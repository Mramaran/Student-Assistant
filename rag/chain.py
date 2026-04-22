"""rag/chain.py — LCEL composition for the UC4 RAG pipeline.

Chain shape:
    retriever -> format_context -> ChatPromptTemplate ->
    ChatGoogleGenerativeAI -> StrOutputParser

``answer(query)`` is the public orchestrator: it retrieves chunks, runs the
LLM, times the wall clock, pulls token usage from ``response.usage_metadata``,
and returns the dict consumed by Streamlit and the test suite.

Reference docs:
  LCEL:                     https://python.langchain.com/docs/concepts/lcel/
  ChatGoogleGenerativeAI:   https://python.langchain.com/docs/integrations/chat/google_generative_ai/
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from core.config import GEMINI_MODEL
from services.escalation import create_ticket as _create_ticket
from services.safety_handler import scrub_answer as _scrub_answer
from verification.confidence import score as _score_confidence
from verification.conflict_detector import detect_conflicts as _detect_conflicts

load_dotenv()


_REPO_ROOT = Path(__file__).resolve().parent.parent
# Test-overridable default; production leaves this None so the escalation
# module uses its own feedback/escalations.jsonl path.
_ESCALATION_LOG_PATH: Optional[Path] = None
_DEFAULT_SYSTEM_PROMPT = (
    "You are a strict academic assistant. Answer only from the provided context "
    "and cite every claim using the [Source N] labels shown in the context. "
    "If the context does not contain the answer, reply exactly: "
    '"The information is not available in the provided sources."'
)


def _read_system_prompt() -> str:
    """Load ``system_prompt.txt`` from the repo root; fall back to a baked default."""
    prompt_path = _REPO_ROOT / "system_prompt.txt"
    if prompt_path.exists():
        text = prompt_path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return _DEFAULT_SYSTEM_PROMPT


def format_context(docs: List[Document]) -> str:
    """Number each retrieved doc as ``[Source N] (filename)`` for citation-friendly prompting."""
    parts: list[str] = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page")
        suffix = f" Pg. {page}" if page is not None else ""
        parts.append(f"[Source {i}] ({source}{suffix}):\n{doc.page_content}")
    return "\n\n".join(parts) if parts else "(no context retrieved)"


_llm_singleton: Optional[ChatGoogleGenerativeAI] = None


def _get_llm() -> ChatGoogleGenerativeAI:
    """Lazy singleton — one ChatGoogleGenerativeAI per process."""
    global _llm_singleton
    if _llm_singleton is None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        _llm_singleton = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=api_key,
            temperature=0.2,
        )
    return _llm_singleton


def _retrieve(query: str, top_k: Optional[int] = None) -> List[Document]:
    from rag.retriever import retrieve_chunks
    return retrieve_chunks(query, top_k=top_k)


def _build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", _read_system_prompt()),
            (
                "human",
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer the question grounded in the context. Cite every claim "
                "using the [Source N] markers shown above. If the answer is not "
                'supported, reply "The information is not available in the '
                'provided sources."',
            ),
        ]
    )


def answer(query: str, top_k: Optional[int] = None) -> dict:
    """Run the LCEL pipeline for ``query`` and return the contract dict.

    Returned keys:
        answer:            str   — generated answer, must cite [Source N]
        sources:           list  — retrieved LangChain Documents
        prompt_tokens:     int   — from ``usage_metadata.input_tokens``
        completion_tokens: int   — from ``usage_metadata.output_tokens``
        latency_ms:        int   — wall clock duration in milliseconds
    """
    start_ns = time.perf_counter_ns()

    docs = _retrieve(query, top_k=top_k)
    conflicts = _detect_conflicts(docs)
    context = format_context(docs)

    prompt = _build_prompt()
    llm = _get_llm()
    parser = StrOutputParser()

    message = llm.invoke(prompt.format_messages(context=context, question=query))
    raw_answer = parser.invoke(message)
    answer_text, redactions = _scrub_answer(raw_answer)

    usage = getattr(message, "usage_metadata", None) or {}
    prompt_tokens = int(usage.get("input_tokens", 0) or 0)
    completion_tokens = int(usage.get("output_tokens", 0) or 0)

    confidence = _score_confidence(answer_text, docs)

    escalation: Optional[dict] = None
    if confidence.get("label") == "Low":
        escalation = _create_ticket(
            query=query,
            sources=[d.metadata.get("source", "unknown") for d in docs if hasattr(d, "metadata")],
            confidence=float(confidence.get("composite", 0.0)),
            verdict={"verdict": "Not Supported"},
            log_path=_ESCALATION_LOG_PATH,
        )

    elapsed_ns = time.perf_counter_ns() - start_ns
    latency_ms = max(1, elapsed_ns // 1_000_000)

    return {
        "answer":            answer_text,
        "sources":           docs,
        "prompt_tokens":     prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_ms":        int(latency_ms),
        "confidence":        confidence,
        "conflicts":         conflicts,
        "escalation":        escalation,
    }

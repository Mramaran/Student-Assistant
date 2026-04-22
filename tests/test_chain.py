"""Task 1.3 / 1.4 — contract tests for the LCEL-based ``rag.chain.answer`` entry,
plus the legacy ``rag.generator.generate_answer`` delegation shim."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from langchain_core.documents import Document

from tests.conftest import skip_on_quota

load_dotenv()

_SKIP_REASON = "GEMINI_API_KEY not set — live-API contract test skipped"
_REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(
    not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
    reason=_SKIP_REASON,
)
@skip_on_quota
def test_answer_returns_contract():
    """``answer`` must return the advertised contract dict."""
    from rag.chain import answer

    resp = answer("What is the minimum attendance percentage?")

    assert isinstance(resp, dict), f"Expected dict, got {type(resp).__name__}"
    assert isinstance(resp.get("answer"), str) and resp["answer"].strip(), (
        "answer must be a non-empty string"
    )

    assert isinstance(resp.get("sources"), list), "sources must be a list"
    for doc in resp["sources"]:
        assert isinstance(doc, Document), (
            f"Each source must be a LangChain Document, got {type(doc).__name__}"
        )

    assert isinstance(resp.get("prompt_tokens"), int) and resp["prompt_tokens"] >= 0
    assert isinstance(resp.get("completion_tokens"), int) and resp["completion_tokens"] >= 0
    assert isinstance(resp.get("latency_ms"), int) and resp["latency_ms"] > 0

    assert "[Source " in resp["answer"], (
        f"Answer must cite at least one [Source N]; got: {resp['answer'][:400]!r}"
    )


def test_generator_no_longer_imports_google_genai():
    """Static check: ``rag/generator.py`` must not import ``google.genai`` anymore."""
    gen_path = _REPO_ROOT / "rag" / "generator.py"
    assert gen_path.exists(), "rag/generator.py missing"
    text = gen_path.read_text(encoding="utf-8")
    for needle in ("from google.genai", "from google import genai", "google.genai.types"):
        assert needle not in text, (
            f"rag/generator.py still references {needle!r}; it should delegate to rag.chain"
        )


@pytest.mark.skipif(
    not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
    reason=_SKIP_REASON,
)
@skip_on_quota
def test_generator_delegates_to_chain():
    """Legacy ``generate_answer`` must return the same contract dict as ``rag.chain.answer``."""
    import rag.generator as gen

    assert hasattr(gen, "generate_answer"), "rag.generator must keep 'generate_answer'"

    resp = gen.generate_answer("What is the minimum attendance percentage?")
    assert isinstance(resp, dict), (
        f"generate_answer must return the chain contract dict; got {type(resp).__name__}"
    )
    for key in ("answer", "sources", "prompt_tokens", "completion_tokens", "latency_ms"):
        assert key in resp, f"Delegated response missing key: {key}"


@pytest.mark.skipif(
    not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
    reason=_SKIP_REASON,
)
@skip_on_quota
def test_answer_includes_confidence():
    """The chain contract must include the 4-factor confidence breakdown (Task 3.2)."""
    from rag.chain import answer

    resp = answer("What is the minimum attendance percentage?")
    assert "confidence" in resp, "answer() must attach the confidence breakdown"

    conf = resp["confidence"]
    assert isinstance(conf, dict)
    assert conf["label"] in {"High", "Medium", "Low"}
    assert isinstance(conf["composite"], float) and 0.0 <= conf["composite"] <= 1.0

    factors = conf["factors"]
    assert set(factors.keys()) == {"cosine", "evidence_alignment", "corroboration", "availability"}

    weights = conf["weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-6


@pytest.mark.skipif(
    not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
    reason=_SKIP_REASON,
)
@skip_on_quota
def test_answer_includes_conflicts_empty_on_aligned_docs(monkeypatch):
    """Task 4.2 — ``conflicts`` must be a list, empty when retrieved docs agree."""
    from rag import chain
    aligned = [
        Document(page_content="Minimum attendance is 75%.", metadata={"source": "a.txt"}),
        Document(page_content="Students need 75% attendance.", metadata={"source": "b.txt"}),
    ]
    monkeypatch.setattr(chain, "_retrieve", lambda q, top_k=None: aligned)

    resp = chain.answer("attendance")
    assert "conflicts" in resp
    assert isinstance(resp["conflicts"], list)
    assert resp["conflicts"] == [], f"Aligned docs should emit no conflicts: {resp['conflicts']}"


@pytest.mark.skipif(
    not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
    reason=_SKIP_REASON,
)
@skip_on_quota
def test_answer_includes_conflicts_populated_on_disagreement(monkeypatch):
    """Task 4.2 — conflicting docs produce at least one ``conflicts`` record."""
    from rag import chain
    conflicting = [
        Document(
            page_content="Minimum attendance requirement is 75%.",
            metadata={"source": "attendance_rules.txt"},
        ),
        Document(
            page_content="Minimum attendance requirement is 80%.",
            metadata={"source": "exam_policy.txt"},
        ),
    ]
    monkeypatch.setattr(chain, "_retrieve", lambda q, top_k=None: conflicting)

    resp = chain.answer("attendance")
    assert "conflicts" in resp
    assert isinstance(resp["conflicts"], list)
    assert len(resp["conflicts"]) >= 1, f"Expected ≥1 conflict, got {resp['conflicts']}"


# ── Task 5.2 — escalation on Low confidence / unsupported verdict ────────────

def _low_confidence_dict() -> dict:
    return {
        "composite": 0.1,
        "factors":   {"cosine": 0.1, "evidence_alignment": 0.0, "corroboration": 0.0, "availability": 0.2},
        "weights":   {"cosine": 0.40, "evidence_alignment": 0.35, "corroboration": 0.15, "availability": 0.10},
        "label":     "Low",
    }


def _high_confidence_dict() -> dict:
    return {
        "composite": 0.85,
        "factors":   {"cosine": 0.95, "evidence_alignment": 0.8, "corroboration": 1.0, "availability": 0.4},
        "weights":   {"cosine": 0.40, "evidence_alignment": 0.35, "corroboration": 0.15, "availability": 0.10},
        "label":     "High",
    }


@pytest.mark.skipif(
    not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
    reason=_SKIP_REASON,
)
@skip_on_quota
def test_low_confidence_surfaces_escalation(monkeypatch, tmp_path):
    """Task 5.2 — Low confidence produces a ticket_id + mailto_url escalation."""
    from rag import chain

    docs = [Document(page_content="Something unrelated.", metadata={"source": "a.txt"})]
    monkeypatch.setattr(chain, "_retrieve", lambda q, top_k=None: docs)
    monkeypatch.setattr(chain, "_score_confidence",
                        lambda answer, documents, verifier_output=None: _low_confidence_dict())

    log_path = tmp_path / "escalations.jsonl"
    monkeypatch.setattr(chain, "_ESCALATION_LOG_PATH", log_path, raising=False)

    resp = chain.answer("something well outside the corpus")
    assert "escalation" in resp
    assert resp["escalation"] is not None, "Low confidence must surface an escalation"
    assert "ticket_id" in resp["escalation"]
    assert resp["escalation"]["mailto_url"].startswith("mailto:")
    assert log_path.exists() and log_path.stat().st_size > 0


@pytest.mark.skipif(
    not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
    reason=_SKIP_REASON,
)
@skip_on_quota
def test_high_confidence_no_escalation(monkeypatch, tmp_path):
    """Task 5.2 — High confidence leaves escalation as None."""
    from rag import chain

    docs = [Document(page_content="Attendance requirement is 75%.", metadata={"source": "a.txt"})]
    monkeypatch.setattr(chain, "_retrieve", lambda q, top_k=None: docs)
    monkeypatch.setattr(chain, "_score_confidence",
                        lambda answer, documents, verifier_output=None: _high_confidence_dict())

    log_path = tmp_path / "escalations.jsonl"
    monkeypatch.setattr(chain, "_ESCALATION_LOG_PATH", log_path, raising=False)

    resp = chain.answer("attendance question")
    assert "escalation" in resp
    assert resp["escalation"] is None, (
        f"High confidence must suppress escalation; got {resp['escalation']!r}"
    )

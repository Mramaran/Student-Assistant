"""verification/confidence.py — 4-factor confidence breakdown for UC4.

Produces a reproducible composite ``confidence`` in [0, 1] from four factors,
each independently bounded in [0, 1]:

  cosine             — sentence-transformer cosine(answer, joined_context)
  evidence_alignment — fraction of proper-noun + numeric anchors extracted
                       from the answer that appear verbatim in the context
  corroboration      — fraction of retrieved sources that contain at least
                       one answer anchor
  availability       — min(n_retrieved / TOP_K, 1.0)

Fixed weights 0.40 / 0.35 / 0.15 / 0.10. The composite is classified via the
``CONF_HIGH`` (0.75) and ``CONF_MEDIUM`` (0.45) thresholds from ``core.config``.

The ``verifier_output`` argument is accepted for call-site back-compat (the
UI and chain already produce it) but is not currently used by the formula.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from langchain_core.documents import Document

from core.config import CONF_HIGH, CONF_MEDIUM, EMBEDDING_MODEL, TOP_K


WEIGHTS: dict[str, float] = {
    "cosine":             0.40,
    "evidence_alignment": 0.35,
    "corroboration":      0.15,
    "availability":       0.10,
}


# ── anchor extraction ────────────────────────────────────────────────────────

_CITATION_RE = re.compile(r"\[Source\s+\d+\][^)\]]*", re.IGNORECASE)
_PROPER_NOUN_RE = re.compile(r"\b(?:[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b")
_NUMBER_ANCHOR_RE = re.compile(
    r"(?:"
    r"\d+(?:\.\d+)?\s*(?:%|percent)"          # percentages
    r"|Rs\.?\s*\d+(?:,\d{3})*(?:\.\d+)?"       # Rupees
    r"|\$\s*\d+(?:,\d{3})*(?:\.\d+)?"          # Dollars
    r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"          # date d/m/y
    r"|(?:19|20)\d{2}"                         # 4-digit year
    r"|\d+(?:\.\d+)?"                          # bare number
    r")",
    re.IGNORECASE,
)

# Tokens that should never count as a distinctive proper-noun anchor.
_STOP_PROPERS: frozenset[str] = frozenset({
    "Source", "The", "A", "An", "I", "You", "We",
    "Sources", "Answer", "Question",
})


def _strip_citations(text: str) -> str:
    return _CITATION_RE.sub(" ", text)


def _extract_anchors(text: str) -> set[str]:
    if not text:
        return set()
    clean = _strip_citations(text)
    propers = {
        m.group(0)
        for m in _PROPER_NOUN_RE.finditer(clean)
        if m.group(0) not in _STOP_PROPERS
    }
    numerics = {m.group(0) for m in _NUMBER_ANCHOR_RE.finditer(clean)}
    return propers | numerics


# ── cosine similarity (lazy-loaded HF embedder) ─────────────────────────────

_embeddings = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def _cosine(a: Iterable[float], b: Iterable[float]) -> float:
    av, bv = list(a), list(b)
    dot = sum(x * y for x, y in zip(av, bv))
    na = sum(x * x for x in av) ** 0.5
    nb = sum(y * y for y in bv) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


# ── factor functions ─────────────────────────────────────────────────────────

def _join_context(docs: list[Document]) -> str:
    return "\n".join(d.page_content for d in docs if d.page_content)


def _cosine_factor(answer: str, context: str) -> float:
    if not answer.strip() or not context.strip():
        return 0.0
    try:
        emb = _get_embeddings()
        return max(0.0, _cosine(emb.embed_query(answer), emb.embed_query(context)))
    except Exception:
        return 0.0


def _evidence_factor(answer: str, context: str) -> float:
    anchors = _extract_anchors(answer)
    if not anchors:
        return 0.0
    lc = context.lower()
    hits = sum(1 for a in anchors if a.lower() in lc)
    return hits / len(anchors)


def _corroboration_factor(answer: str, docs: list[Document]) -> float:
    if not docs:
        return 0.0
    anchors = _extract_anchors(answer)
    if not anchors:
        return 0.0
    supporting = 0
    for doc in docs:
        lc = (doc.page_content or "").lower()
        if any(a.lower() in lc for a in anchors):
            supporting += 1
    return supporting / len(docs)


def _availability_factor(n_docs: int) -> float:
    if TOP_K <= 0:
        return 0.0
    return min(n_docs / TOP_K, 1.0)


# ── classification ───────────────────────────────────────────────────────────

def _classify(composite: float) -> str:
    """Map a composite score to High / Medium / Low per the UC4 thresholds."""
    if composite >= CONF_HIGH:
        return "High"
    if composite >= CONF_MEDIUM:
        return "Medium"
    return "Low"


# ── public entry point ──────────────────────────────────────────────────────

def score(
    answer_text: str,
    retrieved_docs: list[Document],
    verifier_output: Optional[dict] = None,  # accepted for caller back-compat
) -> dict:
    """Return the 4-factor confidence breakdown dict.

    Returned dict:
        composite:  float in [0, 1]
        factors:    {cosine, evidence_alignment, corroboration, availability}
                    — each a float in [0, 1]
        weights:    fixed UC4 weights (sum to 1.0)
        label:      "High" | "Medium" | "Low"
    """
    context = _join_context(retrieved_docs)
    factors: dict[str, float] = {
        "cosine":             float(_cosine_factor(answer_text, context)),
        "evidence_alignment": float(_evidence_factor(answer_text, context)),
        "corroboration":      float(_corroboration_factor(answer_text, retrieved_docs)),
        "availability":       float(_availability_factor(len(retrieved_docs))),
    }
    weights = dict(WEIGHTS)
    composite = sum(factors[k] * weights[k] for k in weights)
    composite = max(0.0, min(1.0, composite))
    return {
        "composite": float(composite),
        "factors":   factors,
        "weights":   weights,
        "label":     _classify(composite),
    }


__all__ = ["WEIGHTS", "score", "_classify"]

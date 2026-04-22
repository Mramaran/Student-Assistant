"""verification/conflict_detector.py — rule-based numeric contradiction detector.

``detect_conflicts(docs)`` scans the retrieved chunks for numeric anchors
(percentages, Rs-denominated amounts, 4-digit years, slash-separated dates)
and groups them by an ``anchor_key`` heuristic: the 3 words immediately
preceding the numeric token, lowercased with English stop words stripped.
When two or more chunks yield the same anchor_key with DIFFERENT values, a
conflict record is emitted.

Each conflict record:
    {
        "anchor_key": str,          # e.g. "attendance requirement"
        "values": [                 # ≥2 entries, each from a distinct doc
            {"source": str, "snippet": str, "value": str},
            ...
        ],
    }
"""

from __future__ import annotations

import re
from typing import Iterable

from langchain_core.documents import Document


# Numeric anchor shapes we care about.
_ANCHOR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\d+(?:\.\d+)?\s*%"),                          # percentage
    re.compile(r"(?:Rs\.?|INR)\s*\d+(?:,\d{3})*(?:\.\d+)?"),    # INR
    re.compile(r"\$\s*\d+(?:,\d{3})*(?:\.\d+)?"),               # USD
    re.compile(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"),               # date d/m/y
    re.compile(r"\b(?:19|20)\d{2}\b"),                          # 4-digit year
)


# Stop words removed from the preceding-words window when building the key.
_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "at", "by", "for", "with", "as", "from",
    "and", "or", "but", "not", "no", "this", "that", "these", "those",
    "it", "its", "their", "our", "his", "her", "your", "my",
    "we", "you", "they", "i",
    "per", "each", "every", "any", "some",
    "rs", "rs.", "inr",
    "must", "shall", "should", "may", "can", "will", "would",
    "have", "has", "had",
})


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")


def _normalize_value(value: str) -> str:
    """Collapse whitespace inside an anchor so "Rs. 25000" and "Rs.  25000" match."""
    return re.sub(r"\s+", " ", value.strip())


def _build_anchor_key(preceding_text: str, target_size: int = 3) -> str:
    """Take up to ``target_size`` non-stopword tokens ending at ``preceding_text``."""
    tokens = _WORD_RE.findall(preceding_text)
    keep: list[str] = []
    for token in reversed(tokens):
        lower = token.lower()
        if lower in _STOP_WORDS:
            continue
        keep.append(lower)
        if len(keep) == target_size:
            break
    keep.reverse()
    return " ".join(keep)


def _snippet_around(text: str, start: int, end: int, window: int = 60) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)
    snippet = text[left:right].strip()
    return re.sub(r"\s+", " ", snippet)


def _iter_anchors(text: str) -> Iterable[tuple[int, int, str]]:
    for pattern in _ANCHOR_PATTERNS:
        for match in pattern.finditer(text):
            yield match.start(), match.end(), match.group(0)


def _extract_anchors_from_doc(doc: Document) -> list[dict]:
    text = doc.page_content or ""
    source = (doc.metadata or {}).get("source", "unknown")
    extracted: list[dict] = []
    for start, end, raw in _iter_anchors(text):
        key = _build_anchor_key(text[:start])
        if not key:
            continue
        extracted.append({
            "anchor_key": key,
            "value":      _normalize_value(raw),
            "snippet":    _snippet_around(text, start, end),
            "source":     source,
        })
    return extracted


def detect_conflicts(docs: list[Document]) -> list[dict]:
    """Return a list of conflict records over ``docs``.

    A conflict is emitted when two or more docs independently assert a value
    under the same anchor key and those values disagree.
    """
    if not docs or len(docs) < 2:
        return []

    grouped: dict[str, list[dict]] = {}
    for doc in docs:
        for entry in _extract_anchors_from_doc(doc):
            grouped.setdefault(entry["anchor_key"], []).append(entry)

    conflicts: list[dict] = []
    for key, entries in grouped.items():
        values = {e["value"] for e in entries}
        sources = {e["source"] for e in entries}
        if len(values) < 2 or len(sources) < 2:
            continue
        # Deduplicate per (source, value) to avoid repeated listings from the
        # same source quoting a number twice.
        seen: set[tuple[str, str]] = set()
        deduped: list[dict] = []
        for entry in entries:
            pair = (entry["source"], entry["value"])
            if pair in seen:
                continue
            seen.add(pair)
            deduped.append({
                "source":  entry["source"],
                "snippet": entry["snippet"],
                "value":   entry["value"],
            })
        if len({e["value"] for e in deduped}) < 2:
            continue
        conflicts.append({"anchor_key": key, "values": deduped})

    return conflicts


__all__ = ["detect_conflicts"]

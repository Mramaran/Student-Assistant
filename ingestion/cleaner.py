"""
Text cleaning helpers for ingestion pipeline.
"""

from __future__ import annotations

import re
from core.guard import MALICIOUS_PATTERNS

_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINES = re.compile(r"\n{3,}")
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_MALICIOUS = [re.compile(p, re.IGNORECASE) for p in MALICIOUS_PATTERNS]

def _normalize_text(text: str) -> str:
    cleaned = _ZERO_WIDTH.sub("", text or "")
    
    # Context Sanitization: Strip malicious prompts from raw documents before chunking
    for pattern in _MALICIOUS:
        cleaned = pattern.sub("[REDACTED_SYSTEM_OVERRIDE]", cleaned)
        
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _MULTI_SPACE.sub(" ", cleaned)
    cleaned = _MULTI_NEWLINES.sub("\n\n", cleaned)
    return cleaned.strip()


def clean_text(docs: list[dict]) -> list[dict]:
    """Normalize text while preserving every metadata key set by the loader."""
    cleaned_docs: list[dict] = []
    for doc in docs:
        text = _normalize_text(doc.get("text", ""))
        if not text:
            continue
        new_doc = dict(doc)  # preserve source, doc_type, checksum, page (if PDF)
        new_doc["text"] = text
        cleaned_docs.append(new_doc)
    return cleaned_docs

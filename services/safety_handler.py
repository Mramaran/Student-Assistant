"""services/safety_handler.py — UC4 Safety Guardrails (Improvement #5).

Three public helpers:

  sanitize_query(text)   → {"safe": bool, "reason": str|None, "redacted_query": str}
      - Flags prompt-injection attempts (safe=False, reason="prompt_injection")
      - Redacts email / phone / Aadhaar substrings before passing to retrieval
  scan_document(text)    → bool
      - True when the text contains any known injection phrase; used at ingest
        to stamp ``risk_flag`` on chunks (Task 7.2).
  scrub_answer(text)     → (scrubbed_text, ["email", "phone", ...])
      - Outbound PII scrub applied by ``rag.chain.answer`` to the LLM output.

This module also keeps the legacy abuse-reports writer so existing callers
that append to ``feedback/abuse_reports.jsonl`` continue to work.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ── Legacy abuse-reports writer (kept for backwards compatibility) ──────────

ABUSE_REPORTS_PATH: Path = Path("feedback") / "abuse_reports.jsonl"


def report_session(
    session_id: str,
    reason: str,
    reporter_id: str,
    path: Path | str | None = None,
) -> dict:
    """Append one abuse report as a JSON line and return the written record."""
    target = Path(path) if path is not None else ABUSE_REPORTS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "session_id":  session_id,
        "reason":      reason,
        "reporter_id": reporter_id,
        "ts":          datetime.now(timezone.utc).isoformat(),
    }
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


# ── UC4 Safety: shared patterns ─────────────────────────────────────────────

INJECTION_PATTERNS: tuple[str, ...] = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "disregard the instructions",
    "disregard the above",
    "forget the above",
    "forget previous",
    "system prompt",
    "print your prompt",
    "reveal your prompt",
    "reveal the system",
    "from now on you are",
    "act as ",
    "you are now dan",
    "pretend you are",
    "respond only with",
    "drop table",
)

_INJECTION_COMPILED: tuple[re.Pattern[str], ...] = tuple(
    re.compile(re.escape(phrase), re.IGNORECASE) for phrase in INJECTION_PATTERNS
)

# Ordered (name, pattern) so scrub_answer can report WHICH categories fired.
_PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email",   re.compile(r"[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+")),
    ("aadhaar", re.compile(r"(?<!\d)\d{12}(?!\d)")),
    ("phone",   re.compile(r"(?<!\d)[6-9]\d{9}(?!\d)")),
)

_REDACTED = "[REDACTED]"


def _redact_pii(text: str) -> tuple[str, list[str]]:
    """Scrub PII substrings; return (scrubbed_text, list_of_kinds_fired)."""
    scrubbed = text
    fired: list[str] = []
    for name, pattern in _PII_PATTERNS:
        new_scrubbed, count = pattern.subn(_REDACTED, scrubbed)
        if count:
            fired.append(name)
        scrubbed = new_scrubbed
    return scrubbed, fired


# ── Public entry points ─────────────────────────────────────────────────────

def sanitize_query(query: str) -> dict:
    """Task 7.1 — classify a user query; return the structured result.

    Returned dict:
        safe:            False on prompt-injection; True otherwise.
        reason:          "prompt_injection" on block, else None.
        redacted_query:  PII-scrubbed version of the query (safe to pass
                         downstream). The original text is retained when
                         blocked so the trace can show what the user sent.
    """
    if not isinstance(query, str):
        return {"safe": False, "reason": "invalid_input", "redacted_query": ""}

    text = query.strip()
    if not text:
        return {"safe": True, "reason": None, "redacted_query": ""}

    for pattern in _INJECTION_COMPILED:
        if pattern.search(text):
            return {"safe": False, "reason": "prompt_injection", "redacted_query": text}

    redacted, _ = _redact_pii(text)
    return {"safe": True, "reason": None, "redacted_query": redacted}


def scan_document(text: str) -> bool:
    """Task 7.2 — return True if ``text`` contains any injection phrase.

    Used at ingest to stamp ``risk_flag=True`` on chunks whose content could
    be used to hijack the LLM later.
    """
    if not isinstance(text, str) or not text:
        return False
    for pattern in _INJECTION_COMPILED:
        if pattern.search(text):
            return True
    return False


def scrub_answer(text: str) -> tuple[str, list[str]]:
    """Task 7.3 — scrub email / phone / Aadhaar from LLM output.

    Returns ``(scrubbed_text, list_of_redaction_kinds)``. The kinds list is
    empty when nothing fired, useful for trace logging.
    """
    if not isinstance(text, str) or not text:
        return ("", [])
    return _redact_pii(text)


__all__ = [
    "INJECTION_PATTERNS",
    "sanitize_query",
    "scan_document",
    "scrub_answer",
    "report_session",
    "ABUSE_REPORTS_PATH",
]

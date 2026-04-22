"""core/trace.py — append-only JSONL trace writer with PII scrubbing.

Every event is stamped with an ISO 8601 UTC ``ts`` and then passed through a
minimal PII scrubber (email, 10-digit phone, 12-digit Aadhaar) before being
appended as a single JSON line to ``feedback/trace.jsonl`` (overridable via
the ``path`` keyword for tests).

Task 7.1 will centralise PII patterns in ``services/safety_handler``; this
module carries its own minimal copy so trace writes never block on that
module's import.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union


DEFAULT_TRACE_PATH = (
    Path(__file__).resolve().parent.parent / "feedback" / "trace.jsonl"
)


# Minimal PII patterns — kept in sync with services.safety_handler (Task 7.1).
_PII_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+"),           # email
    re.compile(r"(?<!\d)\d{12}(?!\d)"),                       # aadhaar (12)
    re.compile(r"(?<!\d)[6-9]\d{9}(?!\d)"),                   # Indian mobile (10)
)
_REDACTED = "[REDACTED]"


def _scrub_string(value: str) -> str:
    scrubbed = value
    for pattern in _PII_PATTERNS:
        scrubbed = pattern.sub(_REDACTED, scrubbed)
    return scrubbed


def _scrub(value: Any) -> Any:
    """Recursively scrub PII patterns out of strings inside dicts / lists."""
    if isinstance(value, str):
        return _scrub_string(value)
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(item) for item in value]
    return value


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(event: dict, path: Union[str, Path, None] = None) -> dict:
    """Append ``event`` as one JSON line to the trace log; return the record.

    The function mutates a copy of ``event`` — the caller's dict is untouched.
    ``ts`` is added last so the caller can supply their own ``ts`` if desired
    (uncommon; the default is the current UTC moment).
    """
    target = Path(path) if path is not None else DEFAULT_TRACE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    record: dict = dict(event)
    record.setdefault("ts", _utc_now_iso())
    record = _scrub(record)

    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record

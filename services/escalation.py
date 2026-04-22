"""services/escalation.py — human-escalation ticket service (UC4 Improvement #3).

``create_ticket`` writes a JSONL ticket record and returns a pre-filled
``mailto:`` URL so the operator can hand the question off to faculty without
leaving the assistant. All ticket metadata is captured for offline review.

The function is pure — it takes a ``log_path`` argument (default:
``feedback/escalations.jsonl`` relative to repo root) so tests can redirect
writes to a temp file without touching the shared log.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence, Union
from urllib.parse import quote

from core.config import ESCALATION_EMAIL


DEFAULT_LOG_PATH = (
    Path(__file__).resolve().parent.parent / "feedback" / "escalations.jsonl"
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_mailto(
    to: str,
    ticket_id: str,
    query: str,
    confidence: float,
    sources: Sequence[str],
    verdict: dict,
) -> str:
    subject = f"[UC4 Escalation {ticket_id}] Needs faculty review"
    body_lines = [
        f"Ticket ID: {ticket_id}",
        f"Query: {query}",
        f"Confidence: {confidence:.2f}",
        f"Verifier: {verdict.get('verdict', 'n/a')}",
        f"Sources: {', '.join(sources) if sources else '(none)'}",
        "",
        "Please review and provide the authoritative answer.",
    ]
    body = "\n".join(body_lines)
    return f"mailto:{to}?subject={quote(subject)}&body={quote(body)}"


def create_ticket(
    query: str,
    sources: Sequence[str],
    confidence: float,
    verdict: dict,
    log_path: Union[str, Path, None] = None,
    to_email: str | None = None,
) -> dict:
    """Append an escalation ticket to JSONL and return a mailto-ready dict.

    Parameters
    ----------
    query, sources, confidence, verdict : captured verbatim in the record
    log_path : optional override for the JSONL file (tests pass tmp_path)
    to_email : optional override for ESCALATION_EMAIL

    Returns
    -------
    {"ticket_id": str, "mailto_url": str, "logged_at": str}
    """
    ticket_id = str(uuid.uuid4())
    target_path = Path(log_path) if log_path is not None else DEFAULT_LOG_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    to = to_email or ESCALATION_EMAIL
    mailto_url = _build_mailto(to, ticket_id, query, confidence, sources, verdict)

    record = {
        "ticket_id":  ticket_id,
        "created_at": _utc_now_iso(),
        "query":      query,
        "sources":    list(sources),
        "confidence": float(confidence),
        "verdict":    verdict,
        "to":         to,
    }
    with target_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "ticket_id":  ticket_id,
        "mailto_url": mailto_url,
        "logged_at":  record["created_at"],
    }


__all__ = ["create_ticket", "DEFAULT_LOG_PATH"]

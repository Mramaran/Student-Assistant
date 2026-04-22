"""Task 5.1 — services.escalation.create_ticket contract."""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import unquote


UUID_HEX_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def test_create_ticket_returns_ticket_id_and_mailto(tmp_path: Path):
    from services.escalation import create_ticket

    log_path = tmp_path / "escalations.jsonl"
    result = create_ticket(
        query="When is the next scholarship deadline?",
        sources=["scholarship_policy.txt"],
        confidence=0.3,
        verdict={"verdict": "Not Supported", "supported": 0, "unsupported": 2},
        log_path=log_path,
    )

    assert isinstance(result, dict)
    assert UUID_HEX_RE.match(result["ticket_id"]), (
        f"ticket_id must be uuid4 string; got {result['ticket_id']!r}"
    )

    mailto = result["mailto_url"]
    assert mailto.startswith("mailto:"), f"mailto_url must start with mailto:; got {mailto!r}"
    # Subject and body should carry the ticket id AND the query (url-encoded).
    decoded = unquote(mailto)
    assert result["ticket_id"] in decoded
    assert "When is the next scholarship deadline?" in decoded


def test_create_ticket_appends_exactly_one_jsonl_line(tmp_path: Path):
    from services.escalation import create_ticket

    log_path = tmp_path / "escalations.jsonl"
    result = create_ticket(
        query="OOD query about weather",
        sources=[],
        confidence=0.1,
        verdict={"verdict": "Not Supported"},
        log_path=log_path,
    )

    lines = [ln for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["ticket_id"] == result["ticket_id"]
    assert parsed["query"] == "OOD query about weather"
    assert parsed["confidence"] == 0.1


def test_create_ticket_appends_multiple_calls(tmp_path: Path):
    from services.escalation import create_ticket

    log_path = tmp_path / "escalations.jsonl"
    ids = {
        create_ticket(query=f"q{i}", sources=[], confidence=0.0, verdict={}, log_path=log_path)["ticket_id"]
        for i in range(3)
    }
    assert len(ids) == 3, "Ticket ids must be unique"

    lines = [ln for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 3

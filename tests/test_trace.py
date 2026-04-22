"""Task 2.1 — core/trace.py writes scrubbed JSONL events."""

from __future__ import annotations

import datetime
import json
from pathlib import Path


def test_trace_appends_single_json_line(tmp_path: Path):
    from core.trace import log

    trace_path = tmp_path / "trace.jsonl"
    log({"event": "answer_rendered", "query": "hello"}, path=trace_path)

    text = trace_path.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) == 1, f"Expected one line, got {len(lines)}"

    parsed = json.loads(lines[0])
    assert parsed["event"] == "answer_rendered"
    assert parsed["query"] == "hello"

    assert "ts" in parsed, "Every event must carry an ISO 8601 UTC 'ts'"
    ts = parsed["ts"]
    datetime.datetime.fromisoformat(ts)
    assert ts.endswith("+00:00") or ts.endswith("Z"), (
        f"Timestamp must be UTC-anchored; got {ts!r}"
    )


def test_trace_redacts_email_pii(tmp_path: Path):
    from core.trace import log

    trace_path = tmp_path / "trace.jsonl"
    log(
        {"event": "query", "query": "Please contact me at foo@bar.com tomorrow"},
        path=trace_path,
    )

    parsed = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
    assert "foo@bar.com" not in parsed["query"], (
        f"Email should be scrubbed; got {parsed['query']!r}"
    )
    assert "[REDACTED]" in parsed["query"]


def test_trace_redacts_phone_and_aadhaar(tmp_path: Path):
    from core.trace import log

    trace_path = tmp_path / "trace.jsonl"
    log(
        {"event": "query", "query": "My phone is 9876543210 and aadhaar 123456789012"},
        path=trace_path,
    )

    parsed = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
    assert "9876543210" not in parsed["query"]
    assert "123456789012" not in parsed["query"]


def test_trace_appends_sequentially(tmp_path: Path):
    from core.trace import log

    trace_path = tmp_path / "trace.jsonl"
    log({"event": "a"}, path=trace_path)
    log({"event": "b"}, path=trace_path)
    log({"event": "c"}, path=trace_path)

    lines = [ln for ln in trace_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 3
    assert [json.loads(ln)["event"] for ln in lines] == ["a", "b", "c"]

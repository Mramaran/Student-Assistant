"""Task 2.4 — feedback/analytics.compute_kpis over the UC4 trace.jsonl schema."""

from __future__ import annotations

import json
from pathlib import Path


EXPECTED_KEYS = {
    "resolution_rate",
    "citation_coverage",
    "avg_confidence",
    "retrieval_hit_rate",
    "escalation_rate",
    "redteam_block_rate",
}


def _write_fixture(path: Path, events: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in events:
            handle.write(json.dumps(record) + "\n")


def test_kpi_keys(tmp_path: Path):
    from feedback.analytics import compute_kpis

    # Ten events: 6 benign queries, 2 escalations, 2 red-team attempts.
    events: list[dict] = []
    for i in range(6):
        events.append({
            "ts": f"2026-04-21T00:00:0{i}+00:00",
            "event": "query",
            "resolved": True,
            "has_citation": True,
            "confidence": 0.8,
            "retrieval_hits": 5,
            "escalated": False,
            "is_redteam": False,
            "blocked": False,
        })
    for i in range(2):
        events.append({
            "ts": f"2026-04-21T00:01:0{i}+00:00",
            "event": "query",
            "resolved": False,
            "has_citation": False,
            "confidence": 0.2,
            "retrieval_hits": 2,
            "escalated": True,
            "is_redteam": False,
            "blocked": False,
        })
    for i in range(2):
        events.append({
            "ts": f"2026-04-21T00:02:0{i}+00:00",
            "event": "query",
            "resolved": False,
            "has_citation": False,
            "confidence": 0.0,
            "retrieval_hits": 0,
            "escalated": False,
            "is_redteam": True,
            "blocked": True,
        })

    fixture = tmp_path / "trace.jsonl"
    _write_fixture(fixture, events)

    kpis = compute_kpis(fixture)
    assert set(kpis.keys()) == EXPECTED_KEYS, (
        f"KPI keys must equal {sorted(EXPECTED_KEYS)}; got {sorted(kpis.keys())}"
    )

    for key, value in kpis.items():
        assert isinstance(value, float), f"{key} must be float, got {type(value).__name__}"
        assert 0.0 <= value <= 1.0, f"{key} must be in [0, 1]; got {value}"

    # Sanity: 6 resolved of 10 total = 0.6
    assert abs(kpis["resolution_rate"] - 0.6) < 1e-9
    # 2 of 2 redteam queries blocked → 1.0
    assert abs(kpis["redteam_block_rate"] - 1.0) < 1e-9
    # 2 of 10 queries escalated → 0.2
    assert abs(kpis["escalation_rate"] - 0.2) < 1e-9


def test_compute_kpis_empty_file(tmp_path: Path):
    from feedback.analytics import compute_kpis

    fixture = tmp_path / "trace.jsonl"
    fixture.write_text("", encoding="utf-8")

    kpis = compute_kpis(fixture)
    assert set(kpis.keys()) == EXPECTED_KEYS
    for value in kpis.values():
        assert value == 0.0, "empty trace → all KPIs must be 0.0"

"""feedback/analytics.py — UC4 KPI computation over ``feedback/trace.jsonl``.

The trace schema (one JSON object per line) is produced by ``core.trace.log``.
Every "query" event is expected to carry:

  resolved:         bool   — did the assistant produce a supported answer?
  has_citation:     bool   — did the rendered answer include at least one
                             [Source N] marker?
  confidence:       float  — 0.0–1.0 composite confidence (Task 3.1)
  retrieval_hits:   int    — number of chunks returned by the retriever
  escalated:        bool   — was a human escalation ticket created?
  is_redteam:       bool   — was this query flagged as an attack attempt?
  blocked:          bool   — was the query refused by the safety layer?

``compute_kpis`` returns exactly six floats in [0.0, 1.0] — the dashboard
surface for the UC4 system.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Union


KPI_KEYS: tuple[str, ...] = (
    "resolution_rate",
    "citation_coverage",
    "avg_confidence",
    "retrieval_hit_rate",
    "escalation_rate",
    "redteam_block_rate",
)


def _iter_events(source: Union[str, Path, Iterable[dict]]) -> list[dict]:
    """Yield event dicts from a JSONL path OR a pre-built iterable."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            return []
        events: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events
    return list(source)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def compute_kpis(source: Union[str, Path, Iterable[dict]]) -> dict[str, float]:
    """Compute the six UC4 KPIs over a ``trace.jsonl`` file or event iterable."""
    events = _iter_events(source)
    queries = [e for e in events if e.get("event") == "query"]
    total = len(queries)

    if total == 0:
        return {key: 0.0 for key in KPI_KEYS}

    resolved = [e for e in queries if e.get("resolved") is True]
    cited = [e for e in resolved if e.get("has_citation") is True]
    hits = [e for e in queries if int(e.get("retrieval_hits") or 0) >= 1]
    escalated = [e for e in queries if e.get("escalated") is True]

    redteam_events = [e for e in queries if e.get("is_redteam") is True]
    blocked_redteam = [e for e in redteam_events if e.get("blocked") is True]

    confidences = [float(e.get("confidence") or 0.0) for e in queries]
    avg_conf = sum(confidences) / total if confidences else 0.0

    return {
        "resolution_rate":     _safe_ratio(len(resolved),        total),
        "citation_coverage":   _safe_ratio(len(cited),           len(resolved)),
        "avg_confidence":      max(0.0, min(1.0, avg_conf)),
        "retrieval_hit_rate":  _safe_ratio(len(hits),            total),
        "escalation_rate":     _safe_ratio(len(escalated),       total),
        "redteam_block_rate":  _safe_ratio(len(blocked_redteam), len(redteam_events)),
    }


__all__ = ["KPI_KEYS", "compute_kpis"]

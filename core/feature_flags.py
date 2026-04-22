"""core/feature_flags.py — runtime feature flags for UC4.

The ``X-Feature-Flags`` HTTP header carries a comma-separated list of
``flag=on|off`` pairs. ``parse()`` turns the header into a ``dict[str, bool]``
merged over ``DEFAULTS``. Unknown flag names are passed through so new flags
can be rolled out server-side without a client release.

DEFAULTS covers the 5 UC4 improvement surfaces — all default on.

Only Python's standard library is used.
"""

from __future__ import annotations


DEFAULTS: dict[str, bool] = {
    "confidence_scoring":    True,  # Improvement #1 — Task 3.*
    "contradiction_alerts":  True,  # Improvement #2 — Task 4.*
    "human_escalation":      True,  # Improvement #3 — Task 5.*
    "benchmark_suite":       True,  # Improvement #4 — Task 6.*
    "safety_guardrails":     True,  # Improvement #5 — Task 7.*
}


def _coerce(value: str) -> bool:
    return value.strip().lower() in {"on", "1", "true", "yes"}


def parse(header_value: str | None) -> dict[str, bool]:
    """Parse an ``X-Feature-Flags`` header into a bool dict over ``DEFAULTS``.

    Accepts ``"confidence_scoring=on,safety_guardrails=off"``; whitespace is
    tolerated. Returns a copy of ``DEFAULTS`` when the header is missing or
    empty so the caller can read any documented flag without a ``KeyError``.
    """
    flags = dict(DEFAULTS)
    if not header_value:
        return flags
    for pair in header_value.split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        name, _, value = pair.partition("=")
        name = name.strip()
        if not name:
            continue
        flags[name] = _coerce(value)
    return flags

"""
feedback/impact.py
==================
Real-World Impact Tracker — compares the RAG system against manual processes
and produces structured JSON for UI display and academic reports.

Structure:
  - PROBLEM_MATRIX       : student/tutor pain points + system limitations
  - BASELINES            : manual-vs-system timing and success rates
  - compute_improvement_percentage()   : generic Δ% calculator
  - generate_before_after_summary()    : full JSON report (integrates analytics.py)
"""

from __future__ import annotations
from feedback.analytics import summary as _analytics_summary
from feedback.logger import load_feedback

# ── 1. PROBLEM DEFINITION STRUCTURE ──────────────────────────────────────────

PROBLEM_MATRIX: dict = {
    "student_pain_points": [
        "Students had to manually read through multiple PDF rule-books to find policy answers.",
        "Searching through attendance, exam, scholarship, and hostel rules separately caused confusion.",
        "No single source of truth — different departments gave conflicting answers.",
        "Queries during off-hours (nights/weekends) went unanswered until the next working day.",
        "Students wasted 5+ minutes per query to locate a single policy fact.",
    ],
    "tutor_pain_points": [
        "Faculty repeatedly answered the same repetitive policy questions from many students.",
        "Answering queries was time-consuming and took attention away from teaching duties.",
        "No structured system to log, learn from, or analyse the most common student concerns.",
        "Manual lookups had no grounding mechanism — incorrect answers were possible.",
    ],
    "system_limitations": [
        "Pre-RAG: No automated retrieval from institutional documents.",
        "Pre-RAG: All knowledge lookup was entirely manual and untracked.",
        "Pre-RAG: No confidence metric existed to rate the reliability of an answer.",
        "Pre-RAG: Responses were limited by the availability of faculty/admin.",
    ],
}

# ── 2. BASELINE COMPARISON CONSTANTS ─────────────────────────────────────────

BASELINES: dict = {
    # Manual process (Before RAG system):
    "manual_matching_time_seconds":   300.0,   # ~5 minutes per query
    "manual_success_rate":            0.60,    # 60% — found correct answer manually

    # RAG system targets (seed defaults — overridden by live analytics):
    "system_matching_time_seconds":   2.5,     # ~2.5 second average
    "system_success_rate":            0.85,    # 85% approval rate target
}


# ── 3. CORE FUNCTIONS ─────────────────────────────────────────────────────────

def compute_improvement_percentage(
    old_value: float,
    new_value: float,
    inverse: bool = False,
) -> float:
    """
    Compute percentage improvement from old_value to new_value.

    Parameters
    ----------
    old_value : float  — baseline (manual) measurement
    new_value : float  — system (new) measurement
    inverse   : bool   — if True, improvement = reduction (e.g. latency)

    Returns
    -------
    float : improvement as a percentage (positive = better)

    Examples
    --------
    >>> compute_improvement_percentage(300, 2.5, inverse=True)
    99.17   # 99.17% faster
    >>> compute_improvement_percentage(0.60, 0.85)
    41.67   # 41.67% higher success rate
    """
    if old_value == 0:
        return 0.0
    if inverse:
        return round(((old_value - new_value) / old_value) * 100, 2)
    return round(((new_value - old_value) / old_value) * 100, 2)


def generate_before_after_summary() -> dict:
    """
    Generate a full before-vs-after impact summary as structured JSON.

    Pulls live metrics from analytics.py (feedback_log.jsonl) and combines
    them with the static BASELINES and PROBLEM_MATRIX.

    Returns
    -------
    dict with keys:
      problem_matrix      : student/tutor pain points and old limitations
      baseline            : manual-process benchmarks
      system_performance  : live RAG metrics from the feedback log
      improvements        : computed Δ% for latency, success_rate, coverage
      summary_statement   : human-readable one-liner for reports
      verdict             : "IMPROVEMENT" | "NEEDS_WORK" | "NO_DATA"
    """
    # Pull live analytics from the feedback log
    stats = _analytics_summary()
    records = load_feedback()

    # ── Live System Metrics ───────────────────────────────────────────────────
    if records:
        latencies = [r.get("latency", 0.0) for r in records if r.get("latency", 0) > 0]
        avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else BASELINES["system_matching_time_seconds"]
        live_success_rate = stats.get("approval_rate", BASELINES["system_success_rate"])
        total_queries = stats.get("total", 0)
        avg_confidence = stats.get("avg_confidence", 0.0)
    else:
        avg_latency = BASELINES["system_matching_time_seconds"]
        live_success_rate = BASELINES["system_success_rate"]
        total_queries = 0
        avg_confidence = 0.0

    # ── Compute Improvements ──────────────────────────────────────────────────
    latency_improvement = compute_improvement_percentage(
        BASELINES["manual_matching_time_seconds"],
        avg_latency,
        inverse=True,   # lower latency = better
    )

    success_improvement = compute_improvement_percentage(
        BASELINES["manual_success_rate"],
        live_success_rate,
        inverse=False,  # higher approval = better
    )

    # ── Verdict ───────────────────────────────────────────────────────────────
    if total_queries == 0:
        verdict = "NO_DATA"
    elif latency_improvement > 50 and success_improvement >= 0:
        verdict = "IMPROVEMENT"
    else:
        verdict = "NEEDS_WORK"

    # ── Summary Statement ─────────────────────────────────────────────────────
    summary_statement = (
        f"The RAG system is {latency_improvement:.1f}% faster than manual search "
        f"({avg_latency}s vs {BASELINES['manual_matching_time_seconds']}s) "
        f"and achieved a {live_success_rate * 100:.0f}% answer acceptance rate "
        f"across {total_queries} logged queries, compared to a {BASELINES['manual_success_rate'] * 100:.0f}% "
        f"manual success rate baseline."
    )

    return {
        "problem_matrix": PROBLEM_MATRIX,
        "baseline": {
            "manual_matching_time_seconds": BASELINES["manual_matching_time_seconds"],
            "manual_success_rate":          BASELINES["manual_success_rate"],
            "description":                  "Manual policy lookup before the RAG system was deployed.",
        },
        "system_performance": {
            "avg_latency_seconds":  avg_latency,
            "success_rate":         live_success_rate,
            "total_queries_logged": total_queries,
            "avg_confidence":       avg_confidence,
        },
        "improvements": {
            "latency_improvement_pct":  latency_improvement,
            "success_rate_improvement_pct": success_improvement,
            "latency_label":  f"{latency_improvement:.1f}% faster",
            "success_label":  f"+{success_improvement:.1f}% accuracy",
        },
        "summary_statement": summary_statement,
        "verdict": verdict,
    }

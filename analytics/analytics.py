"""
feedback/analytics.py  (v2.0 — Advanced Evaluation + A/B Testing)
===================================================================
Reads feedback_log.jsonl and computes a full evaluation suite:

CORE METRICS
  summary()                     → overall stats + user_satisfaction_index
  match_success_rate()          → % of sessions that yielded a confident answer
  session_completion_rate()     → % of sessions that received a rating
  average_feedback_score()      → numeric score from up/down/neutral ratings

A/B TESTING
  compare_random_vs_smart_matching() → synthetic A/B split based on confidence
  get_experiment_results()           → retrieve stored experiment logs

ML-STYLE EVALUATION
  compute_precision_recall_f1()  → precision / recall / F1 over verified answers

TREND TRACKING
  session_success_trend()        → rolling confidence over time
  user_engagement_trend()        → query volume per day

All outputs are structured dicts ready for dashboard visualization.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime
from feedback.logger import load_feedback

EXPERIMENT_LOG = os.path.join("feedback", "ab_experiments.jsonl")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_experiment(name: str, results: dict) -> None:
    """Persist an A/B experiment result to the experiment log."""
    os.makedirs("feedback", exist_ok=True)
    record = {"timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
              "experiment": name, **results}
    with open(EXPERIMENT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_experiments() -> list[dict]:
    """Load all stored experiment results."""
    if not os.path.exists(EXPERIMENT_LOG):
        return []
    results = []
    with open(EXPERIMENT_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return results


def _rating_to_score(rating: str) -> float:
    """Map rating string to numeric score."""
    return {"up": 1.0, "neutral": 0.5, "down": 0.0}.get(rating, 0.5)


# ── Core Metrics ──────────────────────────────────────────────────────────────

def summary() -> dict:
    """Return high-level feedback statistics including latency and verification rate."""
    records = load_feedback()
    if not records:
        return {
            "total": 0, "thumbs_up": 0, "thumbs_down": 0,
            "avg_confidence": 0.0, "avg_latency": 0.0, "verified_rate": 0.0,
            "approval_rate": 0.0, "user_satisfaction_index": 0.0,
        }

    up    = sum(1 for r in records if r.get("rating") == "up")
    down  = sum(1 for r in records if r.get("rating") == "down")
    avg_conf = round(sum(r.get("confidence", 0) for r in records) / len(records), 3)

    latencies = [r.get("latency", 0.0) for r in records if r.get("latency", 0) > 0]
    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

    verified_count = sum(1 for r in records if r.get("verified", False))
    verified_rate = round(verified_count / len(records), 2) if records else 0.0

    # User Satisfaction Index = weighted blend of rating + confidence
    scores = [_rating_to_score(r.get("rating", "neutral")) for r in records]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    usi = round((avg_score * 0.6) + (avg_conf * 0.4), 3)

    return {
        "total":                   len(records),
        "thumbs_up":               up,
        "thumbs_down":             down,
        "avg_confidence":          avg_conf,
        "avg_latency":             avg_latency,
        "verified_rate":           verified_rate,
        "approval_rate":           round(up / len(records), 2) if records else 0.0,
        "user_satisfaction_index": usi,
    }


def match_success_rate(confidence_threshold: float = 0.5) -> dict:
    """
    Percentage of queries where the system returned a confident answer.
    'Successful' = confidence >= threshold AND not rejected.
    """
    records = load_feedback()
    if not records:
        return {"total": 0, "successful": 0, "match_success_rate": 0.0}

    successful = sum(
        1 for r in records
        if r.get("confidence", 0.0) >= confidence_threshold
    )
    rate = round(successful / len(records), 3)
    return {
        "total":              len(records),
        "successful":         successful,
        "match_success_rate": rate,
        "threshold_used":     confidence_threshold,
    }


def session_completion_rate() -> dict:
    """
    Percentage of queries that received an explicit user rating (up/down).
    Neutral = implicit (auto-logged), not counted as completed.
    """
    records = load_feedback()
    if not records:
        return {"total": 0, "rated": 0, "session_completion_rate": 0.0}

    rated = sum(1 for r in records if r.get("rating") in ("up", "down"))
    rate = round(rated / len(records), 3)
    return {
        "total":                   len(records),
        "rated":                   rated,
        "session_completion_rate": rate,
    }


def average_feedback_score() -> dict:
    """
    Numeric feedback score: up=1.0, neutral=0.5, down=0.0.
    Returns mean score and distribution.
    """
    records = load_feedback()
    if not records:
        return {"avg_score": 0.0, "distribution": {}}

    scores = [_rating_to_score(r.get("rating", "neutral")) for r in records]
    dist = {
        "up":      sum(1 for r in records if r.get("rating") == "up"),
        "neutral": sum(1 for r in records if r.get("rating") == "neutral"),
        "down":    sum(1 for r in records if r.get("rating") == "down"),
    }
    return {
        "avg_score":    round(sum(scores) / len(scores), 3),
        "distribution": dist,
        "total":        len(records),
    }


# ── A/B Testing ───────────────────────────────────────────────────────────────

def compare_random_vs_smart_matching(save: bool = True) -> dict:
    """
    Synthetic A/B experiment:
      Group A ('random')  = bottom 50% confidence records (simulates random retrieval)
      Group B ('smart')   = top 50% confidence records (RAG system's smart matching)

    Compares approval rate, avg confidence, and avg latency between groups.
    Results are saved to ab_experiments.jsonl for longitudinal tracking.
    """
    records = load_feedback()
    if len(records) < 4:
        return {"error": "Insufficient data. Need at least 4 logged queries."}

    sorted_records = sorted(records, key=lambda r: r.get("confidence", 0.0))
    mid = len(sorted_records) // 2

    group_a = sorted_records[:mid]   # lower half  → "random" baseline
    group_b = sorted_records[mid:]   # upper half  → "smart" (RAG system)

    def _group_stats(group: list[dict]) -> dict:
        if not group:
            return {}
        up = sum(1 for r in group if r.get("rating") == "up")
        confs = [r.get("confidence", 0.0) for r in group]
        lats  = [r.get("latency", 0.0) for r in group if r.get("latency", 0) > 0]
        return {
            "count":         len(group),
            "approval_rate": round(up / len(group), 3),
            "avg_confidence": round(sum(confs) / len(confs), 3),
            "avg_latency":    round(sum(lats) / len(lats), 2) if lats else 0.0,
        }

    a_stats = _group_stats(group_a)
    b_stats = _group_stats(group_b)

    conf_lift = round(b_stats["avg_confidence"] - a_stats["avg_confidence"], 3)
    rate_lift = round(b_stats["approval_rate"]  - a_stats["approval_rate"],  3)

    result = {
        "experiment":  "random_vs_smart_matching",
        "group_A_random": a_stats,
        "group_B_smart":  b_stats,
        "lift": {
            "confidence_lift": conf_lift,
            "approval_rate_lift": rate_lift,
            "winner": "B_smart" if conf_lift >= 0 else "A_random",
        },
        "recommendation": (
            "Smart RAG matching outperforms random retrieval."
            if conf_lift >= 0
            else "Insufficient signal — needs more data."
        ),
    }

    if save:
        _save_experiment("random_vs_smart_matching", result)

    return result


def get_experiment_results(experiment_name: str | None = None) -> list[dict]:
    """
    Retrieve stored A/B experiment results.
    Optionally filter by experiment name.
    """
    all_exps = _load_experiments()
    if experiment_name:
        return [e for e in all_exps if e.get("experiment") == experiment_name]
    return all_exps


# ── ML-Style Evaluation ───────────────────────────────────────────────────────

def compute_precision_recall_f1(confidence_threshold: float = 0.5) -> dict:
    """
    ML-style evaluation treating the system as a binary classifier:
      Positive class = answer is 'verified' (ground truth)
      Predicted positive = confidence >= threshold

    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F1        = 2 * P * R / (P + R)
    """
    records = load_feedback()
    if not records:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "support": 0}

    tp = sum(1 for r in records
             if r.get("verified") and r.get("confidence", 0) >= confidence_threshold)
    fp = sum(1 for r in records
             if not r.get("verified") and r.get("confidence", 0) >= confidence_threshold)
    fn = sum(1 for r in records
             if r.get("verified") and r.get("confidence", 0) < confidence_threshold)

    precision = round(tp / (tp + fp), 3) if (tp + fp) > 0 else 0.0
    recall    = round(tp / (tp + fn), 3) if (tp + fn) > 0 else 0.0
    f1        = round(2 * precision * recall / (precision + recall), 3) \
                if (precision + recall) > 0 else 0.0

    return {
        "precision":         precision,
        "recall":            recall,
        "f1_score":          f1,
        "support":           len(records),
        "true_positives":    tp,
        "false_positives":   fp,
        "false_negatives":   fn,
        "threshold_used":    confidence_threshold,
    }


# ── Trend Tracking ────────────────────────────────────────────────────────────

def session_success_trend(window: int = 10) -> list[dict]:
    """
    Rolling average confidence over the last N sessions.
    Returns a time-series list ready for chart visualization.
    """
    records = load_feedback()
    if not records:
        return []

    trend = []
    for i in range(len(records)):
        start = max(0, i - window + 1)
        window_records = records[start: i + 1]
        confs = [r.get("confidence", 0.0) for r in window_records]
        avg_conf = round(sum(confs) / len(confs), 3) if confs else 0.0
        trend.append({
            "index":           i + 1,
            "timestamp":       records[i].get("timestamp", ""),
            "rolling_confidence": avg_conf,
            "query_preview":   records[i].get("query", "")[:40] + "...",
        })
    return trend


def user_engagement_trend() -> list[dict]:
    """
    Daily query volume and average confidence.
    Returns a time-series list grouped by date, ready for bar/line charts.
    """
    records = load_feedback()
    if not records:
        return []

    daily: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        ts = r.get("timestamp", "")
        date = ts[:10] if len(ts) >= 10 else "unknown"
        daily[date].append(r)

    trend = []
    for date in sorted(daily.keys()):
        day_records = daily[date]
        confs = [r.get("confidence", 0.0) for r in day_records]
        up = sum(1 for r in day_records if r.get("rating") == "up")
        trend.append({
            "date":           date,
            "query_count":    len(day_records),
            "avg_confidence": round(sum(confs) / len(confs), 3) if confs else 0.0,
            "thumbs_up":      up,
        })
    return trend


# ── Existing Functions (Preserved) ────────────────────────────────────────────

def failing_queries(top_n: int = 10) -> list[dict]:
    """Return the most frequently 👎-rated query patterns."""
    records = load_feedback()
    counts: dict[str, int] = defaultdict(int)
    for r in records:
        if r.get("rating") == "down":
            key = r.get("query", "").lower().strip()
            counts[key] += 1
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [{"query": q, "count": c} for q, c in sorted_items[:top_n]]


def source_stats() -> list[dict]:
    """Return average confidence broken down by source document."""
    records = load_feedback()
    source_conf: dict[str, list[float]] = defaultdict(list)
    for r in records:
        for src in r.get("sources", []):
            source_conf[src].append(r.get("confidence", 0.0))
    result = []
    for src, confs in source_conf.items():
        result.append({
            "source":      src,
            "avg_conf":    round(sum(confs) / len(confs), 3),
            "num_answers": len(confs),
        })
    return sorted(result, key=lambda x: x["avg_conf"])


def low_confidence_failures(threshold: float = 0.5) -> list[dict]:
    """Return 👎-rated answers that also had low confidence — most actionable failures."""
    records = load_feedback()
    return [
        r for r in records
        if r.get("rating") == "down" and r.get("confidence", 1.0) < threshold
    ]

"""
Feedback Logger
================
Stores per-answer thumbs-up / thumbs-down ratings in a JSONL file.

Each line is a JSON record:
  {
    "timestamp":  "2026-04-03T08:30:00",
    "query":      "what is the attendance requirement?",
    "answer":     "The minimum attendance requirement is 75%...",
    "latency":    4.25,
    "retrieval_hits": 5,
    "token_usage": 150,
    "verified":   true,
    "rating":     "up",             # "up" | "down"
    "sources":    ["attendance_rules.txt", "CIT_Academic_Calendar.txt"]
  }

This log is the foundation for:
  - Manual review of poor answers
  - Future RLHF / fine-tuning
  - Auto-evaluation dashboards (see analytics.py)
"""

from __future__ import annotations
import json
import os
from datetime import datetime

FEEDBACK_DIR    = "feedback"
FEEDBACK_FILE   = os.path.join(FEEDBACK_DIR, "feedback_log.jsonl")
TUTOR_FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, "tutor_feedback_log.jsonl")


def log_feedback(
    query: str,
    answer: str,
    confidence: float,
    verified: bool,
    rating: str,          # "up" or "down"
    latency: float = 0.0,
    retrieval_hits: int = 0,
    token_usage: int = 0,
    sources: list[str] | None = None,
) -> None:
    """
    Append a feedback record to the JSONL log.

    Parameters
    ----------
    query      : str   — original user query
    answer     : str   — generated answer
    confidence : float — confidence score (0.0–1.0)
    verified   : bool  — verifier result
    rating     : str   — "up" or "down"
    sources    : list  — source filenames used for the answer
    """
    os.makedirs(FEEDBACK_DIR, exist_ok=True)

    record = {
        "timestamp":  datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "query":      query,
        "answer":     answer[:500],
        "confidence": round(float(confidence or 0.0), 3),
        "latency":    round(latency, 2),
        "retrieval_hits": retrieval_hits,
        "token_usage":    token_usage,
        "verified":   verified,
        "rating":     rating,
        "sources":    sources or [],
    }

    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    emoji = "👍" if rating == "up" else "👎"
    print(f"📝 Feedback logged: {emoji}  query='{query[:40]}...'")


def log_tutor_feedback(
    tutor_id: str,
    student_id: str,
    session_quality: float,      # 0.0 – 1.0  (overall session quality)
    tutor_behavior: float,       # 0.0 – 1.0  (punctuality, respect, engagement)
    concept_clarity: float,      # 0.0 – 1.0  (how clearly tutor explained)
    comment: str = "",
    rating: str = "neutral",     # "up" | "neutral" | "down"
) -> None:
    """
    Append a structured tutor session feedback record.
    Used by update_tutor_ranking() in feedback/tutor_ranker.py.
    """
    os.makedirs(FEEDBACK_DIR, exist_ok=True)

    # Composite score: weighted blend of the three categories
    composite = round(
        0.40 * session_quality +
        0.30 * tutor_behavior  +
        0.30 * concept_clarity,
        3
    )

    record = {
        "timestamp":       datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "tutor_id":        tutor_id,
        "student_id":      student_id,
        "session_quality": round(session_quality, 3),
        "tutor_behavior":  round(tutor_behavior, 3),
        "concept_clarity": round(concept_clarity, 3),
        "composite_score": composite,
        "comment":         comment[:300],
        "rating":          rating,
    }

    with open(TUTOR_FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    emoji = "👍" if rating == "up" else ("👎" if rating == "down" else "➖")
    print(f"📝 Tutor feedback logged: {emoji}  tutor={tutor_id}  composite={composite}")


def load_feedback() -> list[dict]:
    """Load all RAG feedback records from the JSONL log."""
    if not os.path.exists(FEEDBACK_FILE):
        return []
    records = []
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def load_tutor_feedback() -> list[dict]:
    """Load all tutor session feedback records."""
    if not os.path.exists(TUTOR_FEEDBACK_FILE):
        return []
    records = []
    with open(TUTOR_FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records

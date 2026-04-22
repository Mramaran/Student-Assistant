"""
core/edge_handler.py — Edge Case Engine
========================================
Handles all real-world failure modes in the tutor matching lifecycle.
Returns structured fallback responses — no silent failures.

SCENARIOS HANDLED
  1. no_tutor_available       → suggest closest partial match
  2. tutor_cancels            → auto-reassign next best from ranked pool
  3. student_cancels          → reschedule options
  4. repeated_no_shows        → flag user, restrict bookings
  5. low_rated_tutor          → deprioritize in next match cycle
  6. time_overlap_conflict    → detect and suggest alternative slot
  7. last_minute_booking_fail → fallback to async / recorded session
  8. simultaneous_requests    → queue overflow handling

PUBLIC API
  handle(event_type, context)  → structured fallback response dict
"""

from __future__ import annotations
from datetime import datetime
from typing import Any
import json, os

# ── Constants ─────────────────────────────────────────────────────────────────

NO_SHOW_FLAG_THRESHOLD = 3          # flag after N no-shows
LOW_RATING_THRESHOLD   = 0.35       # TutorProfile.popularity_score below this
LAST_MINUTE_MINUTES    = 30         # booking within 30 min = last-minute
FLAG_LOG_FILE          = os.path.join("feedback", "flagged_users.jsonl")

VALID_EVENTS = {
    "no_tutor_available",
    "tutor_cancels",
    "student_cancels",
    "repeated_no_shows",
    "low_rated_tutor",
    "time_overlap_conflict",
    "last_minute_booking_fail",
    "simultaneous_requests",
    "spam_detected",
    "repeated_cancellations",
}


# ── Internal Helpers ──────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _response(
    event: str,
    status: str,
    action: str,
    message: str,
    data: dict | None = None,
) -> dict:
    """Build a standardised fallback response."""
    return {
        "timestamp":  _ts(),
        "event_type": event,
        "status":     status,       # "resolved" | "escalated" | "flagged" | "queued"
        "action":     action,
        "message":    message,
        "data":       data or {},
    }


def _log_flag(user_id: str, reason: str, count: int) -> None:
    """Persist a user flag to the flag log."""
    os.makedirs("feedback", exist_ok=True)
    record = {
        "timestamp": _ts(),
        "user_id":   user_id,
        "reason":    reason,
        "count":     count,
    }
    with open(FLAG_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _load_flags() -> list[dict]:
    if not os.path.exists(FLAG_LOG_FILE):
        return []
    records = []
    with open(FLAG_LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


# ── Edge Case Handlers ────────────────────────────────────────────────────────

def _handle_no_tutor_available(ctx: dict) -> dict:
    """
    No tutor matches the student's exact criteria.
    Strategy: suggest the closest partial match from candidates.

    ctx keys: student_id, subjects, candidates (list of TutorProfile-like dicts)
    """
    candidates = ctx.get("candidates", [])
    student_subjects = {s.lower() for s in ctx.get("subjects", [])}

    if not candidates:
        return _response(
            event="no_tutor_available",
            status="escalated",
            action="notify_admin",
            message="No tutors available. Admin has been notified.",
            data={"student_id": ctx.get("student_id"), "subjects": list(student_subjects)},
        )

    # Find closest partial match by skill overlap
    def _overlap(c):
        skills = {s.lower() for s in c.get("skills", [])}
        return len(student_subjects & skills)

    closest = max(candidates, key=_overlap)

    return _response(
        event="no_tutor_available",
        status="resolved",
        action="suggest_closest_match",
        message=(
            f"No exact match found. Closest available tutor is "
            f"{closest.get('name', 'Unknown')} with partial subject overlap."
        ),
        data={
            "suggested_tutor": closest.get("tutor_id"),
            "tutor_name":      closest.get("name"),
            "overlap_subjects": list(
                student_subjects & {s.lower() for s in closest.get("skills", [])}
            ),
        },
    )


def _handle_tutor_cancels(ctx: dict) -> dict:
    """
    Assigned tutor cancels the session.
    Strategy: auto-assign the next-best tutor from ranked_tutors list.

    ctx keys: cancelled_tutor_id, ranked_tutors (list of scored dicts), session_id
    """
    cancelled = ctx.get("cancelled_tutor_id", "")
    ranked    = [t for t in ctx.get("ranked_tutors", []) if t.get("tutor_id") != cancelled]

    if not ranked:
        return _response(
            event="tutor_cancels",
            status="escalated",
            action="notify_admin",
            message="Cancellation: no replacement tutor available. Admin notified.",
            data={"cancelled_tutor_id": cancelled, "session_id": ctx.get("session_id")},
        )

    replacement = ranked[0]
    return _response(
        event="tutor_cancels",
        status="resolved",
        action="auto_reassign",
        message=(
            f"Tutor {cancelled} cancelled. Auto-assigned replacement: "
            f"{replacement.get('name', '?')} ({replacement.get('confidence', 0)}% match confidence)."
        ),
        data={
            "original_tutor_id":    cancelled,
            "replacement_tutor_id": replacement.get("tutor_id"),
            "replacement_name":     replacement.get("name"),
            "match_confidence":     replacement.get("confidence"),
            "session_id":           ctx.get("session_id"),
        },
    )


def _handle_student_cancels(ctx: dict) -> dict:
    """
    Student cancels their booked session.
    Strategy: offer 3 reschedule slot options from the tutor's availability.

    ctx keys: student_id, tutor_id, tutor_availability (list of slot strings), session_id
    """
    slots = ctx.get("tutor_availability", [])
    options = slots[:3] if len(slots) >= 3 else slots

    return _response(
        event="student_cancels",
        status="resolved",
        action="offer_reschedule",
        message=(
            f"Session cancelled by student {ctx.get('student_id', '?')}. "
            f"{len(options)} reschedule options offered."
        ),
        data={
            "student_id":        ctx.get("student_id"),
            "tutor_id":          ctx.get("tutor_id"),
            "session_id":        ctx.get("session_id"),
            "reschedule_options": options,
        },
    )


def _handle_repeated_no_shows(ctx: dict) -> dict:
    """
    User (student or tutor) accumulates repeated no-shows.
    Strategy: flag after threshold; restrict future bookings.

    ctx keys: user_id, no_show_count, user_type ("student"|"tutor")
    """
    user_id   = ctx.get("user_id", "")
    count     = ctx.get("no_show_count", 0)
    user_type = ctx.get("user_type", "student")

    if count >= NO_SHOW_FLAG_THRESHOLD:
        _log_flag(user_id, f"repeated_no_shows ({user_type})", count)
        return _response(
            event="repeated_no_shows",
            status="flagged",
            action="restrict_bookings",
            message=(
                f"{user_type.capitalize()} {user_id} has {count} no-shows "
                f"(≥{NO_SHOW_FLAG_THRESHOLD}). Booking restricted pending review."
            ),
            data={
                "user_id":         user_id,
                "user_type":       user_type,
                "no_show_count":   count,
                "threshold":       NO_SHOW_FLAG_THRESHOLD,
                "action_required": "admin_review",
            },
        )

    remaining = NO_SHOW_FLAG_THRESHOLD - count
    return _response(
        event="repeated_no_shows",
        status="resolved",
        action="warn_user",
        message=(
            f"{user_type.capitalize()} {user_id} has {count} no-show(s). "
            f"Warning issued. {remaining} more will trigger a restriction."
        ),
        data={"user_id": user_id, "no_show_count": count, "remaining_before_flag": remaining},
    )


def _handle_low_rated_tutor(ctx: dict) -> dict:
    """
    Tutor's feedback-derived popularity_score is below threshold.
    Strategy: deprioritize by reducing popularity_score, log for review.

    ctx keys: tutor_id, tutor_name, current_score, tutors (list — mutated in-place)
    """
    tutor_id  = ctx.get("tutor_id", "")
    name      = ctx.get("tutor_name", tutor_id)
    score     = ctx.get("current_score", 0.0)
    tutors    = ctx.get("tutors", [])   # list of TutorProfile objects

    updated = False
    for t in tutors:
        if getattr(t, "tutor_id", None) == tutor_id:
            old = t.popularity_score
            t.popularity_score = max(0.05, t.popularity_score * 0.75)
            updated = True
            break

    return _response(
        event="low_rated_tutor",
        status="resolved",
        action="deprioritize_tutor",
        message=(
            f"Tutor {name} has low rating ({score:.2f} < {LOW_RATING_THRESHOLD}). "
            f"Deprioritized for upcoming matches."
        ),
        data={
            "tutor_id":    tutor_id,
            "old_score":   score,
            "score_floor": 0.05,
            "in_pool_updated": updated,
        },
    )


def _handle_time_overlap_conflict(ctx: dict) -> dict:
    """
    Requested slot conflicts with an existing booking.
    Strategy: suggest the next available non-conflicting slot.

    ctx keys: student_id, requested_slot, booked_slots (list), available_slots (list)
    """
    requested       = ctx.get("requested_slot", "")
    booked          = set(ctx.get("booked_slots", []))
    all_available   = ctx.get("available_slots", [])
    alternatives    = [s for s in all_available if s not in booked and s != requested]

    if not alternatives:
        return _response(
            event="time_overlap_conflict",
            status="escalated",
            action="notify_admin",
            message=f"Slot '{requested}' conflicts and no alternatives available.",
            data={"student_id": ctx.get("student_id"), "requested_slot": requested},
        )

    return _response(
        event="time_overlap_conflict",
        status="resolved",
        action="suggest_alternative_slots",
        message=f"Slot '{requested}' is taken. {len(alternatives[:3])} alternatives suggested.",
        data={
            "student_id":      ctx.get("student_id"),
            "conflicted_slot": requested,
            "alternatives":    alternatives[:3],
        },
    )


def _handle_last_minute_booking_fail(ctx: dict) -> dict:
    """
    Booking attempted within LAST_MINUTE_MINUTES of session start.
    Strategy: fallback to async/recorded session or waitlist.

    ctx keys: student_id, tutor_id, minutes_until_session
    """
    minutes = ctx.get("minutes_until_session", 0)
    return _response(
        event="last_minute_booking_fail",
        status="resolved",
        action="offer_async_fallback",
        message=(
            f"Booking attempt {minutes} min before session start "
            f"(threshold: {LAST_MINUTE_MINUTES} min). "
            f"Offering async recorded session or waitlist slot."
        ),
        data={
            "student_id":           ctx.get("student_id"),
            "tutor_id":             ctx.get("tutor_id"),
            "minutes_until_session": minutes,
            "fallback_options": ["recorded_session", "waitlist", "next_available_slot"],
        },
    )


def _handle_simultaneous_requests(ctx: dict) -> dict:
    """
    Multiple students request the same tutor simultaneously.
    Strategy: queue requests by priority (urgency + timestamp).

    ctx keys: tutor_id, requests (list of {student_id, urgency, timestamp})
    """
    requests = ctx.get("requests", [])
    tutor_id = ctx.get("tutor_id", "")

    # Sort: urgent first, then by earliest timestamp
    sorted_requests = sorted(
        requests,
        key=lambda r: (not r.get("urgency", False), r.get("timestamp", "9999"))
    )

    confirmed     = sorted_requests[0] if sorted_requests else {}
    waitlisted    = sorted_requests[1:]

    return _response(
        event="simultaneous_requests",
        status="queued",
        action="queue_by_priority",
        message=(
            f"{len(requests)} simultaneous requests for tutor {tutor_id}. "
            f"Student {confirmed.get('student_id', '?')} confirmed. "
            f"{len(waitlisted)} waitlisted."
        ),
        data={
            "tutor_id":   tutor_id,
            "confirmed":  confirmed,
            "waitlisted": waitlisted,
        },
    )


# ── Abuse Detection ───────────────────────────────────────────────────────────

ABUSE_THRESHOLDS = {
    "spam_limit": 10,        # requests per minute
    "cancel_limit": 5,      # cancellations per week
}

def handle_abuse(event_type: str, context: dict) -> dict:
    """
    Handle security and abuse edge cases.
    """
    user_id = context.get("user_id", "ANON")
    
    if event_type == "spam_detected":
        count = context.get("request_count", 0)
        if count > ABUSE_THRESHOLDS["spam_limit"]:
            return {
                "status": "flagged",
                "action": "rate_limit_user",
                "message": f"User {user_id} flagged for spamming {count} requests.",
                "data": {"user_id": user_id, "limit": ABUSE_THRESHOLDS["spam_limit"]}
            }
    
    if event_type == "repeated_cancellations":
        count = context.get("cancel_count", 0)
        if count > ABUSE_THRESHOLDS["cancel_limit"]:
            return {
                "status": "restricted",
                "action": "suspend_booking_privileges",
                "message": f"User {user_id} suspended due to {count} cancellations.",
                "data": {"user_id": user_id, "count": count}
            }
            
    return {"status": "normal", "action": "none"}

# ── Entry Point ───────────────────────────────────────────────────────────────

def handle(event_type: str, context: dict | None = None) -> dict:
    """
      timestamp, event_type, status, action, message, data
    """
    context = context or {}

    if event_type in ["spam_detected", "repeated_cancellations"]:
        return handle_abuse(event_type, context)

    _HANDLERS = {
        "no_tutor_available":       _handle_no_tutor_available,
        "tutor_cancels":            _handle_tutor_cancels,
        "student_cancels":          _handle_student_cancels,
        "repeated_no_shows":        _handle_repeated_no_shows,
        "low_rated_tutor":          _handle_low_rated_tutor,
        "time_overlap_conflict":    _handle_time_overlap_conflict,
        "last_minute_booking_fail": _handle_last_minute_booking_fail,
        "simultaneous_requests":    _handle_simultaneous_requests,
    }

    if event_type not in VALID_EVENTS:
        return _response(
            event=event_type,
            status="error",
            action="none",
            message=f"Unknown event_type '{event_type}'. Valid: {sorted(VALID_EVENTS)}",
        )

    handler = _HANDLERS.get(event_type)
    if not handler:
         return _response(event=event_type, status="error", action="none", message="Handler missing")
    try:
        return handler(context)
    except Exception as exc:
        return _response(
            event=event_type,
            status="error",
            action="none",
            message=f"Handler crashed: {type(exc).__name__}: {exc}",
            data={"context_keys": list(context.keys())},
        )


def get_flagged_users() -> list[dict]:
    """Return all flagged users from the persistent log."""
    return _load_flags()

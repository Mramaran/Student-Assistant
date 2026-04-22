"""Task 4.1 — anchor-based contradiction detection."""

from __future__ import annotations

from langchain_core.documents import Document


def _doc(text: str, source: str) -> Document:
    return Document(page_content=text, metadata={"source": source})


def test_conflicting_attendance_detected():
    from verification.conflict_detector import detect_conflicts

    a = _doc("Minimum attendance requirement is 75%.", "attendance_rules.txt")
    b = _doc("Minimum attendance requirement is 80%.", "exam_policy.txt")

    conflicts = detect_conflicts([a, b])
    assert isinstance(conflicts, list)
    assert len(conflicts) == 1, f"Expected exactly one conflict, got {len(conflicts)}: {conflicts}"

    conflict = conflicts[0]
    assert "anchor_key" in conflict
    assert "values" in conflict
    assert isinstance(conflict["anchor_key"], str) and conflict["anchor_key"]

    values = conflict["values"]
    assert isinstance(values, list) and len(values) == 2
    value_set = {v["value"] for v in values}
    assert value_set == {"75%", "80%"}, f"Expected {{'75%', '80%'}}, got {value_set}"
    for entry in values:
        assert set(entry.keys()) >= {"source", "snippet", "value"}
        assert entry["source"] in {"attendance_rules.txt", "exam_policy.txt"}


def test_aligned_docs_emit_nothing():
    from verification.conflict_detector import detect_conflicts

    a = _doc("Minimum attendance requirement is 75%.", "rules_a.txt")
    b = _doc("Students must maintain at least 75% attendance.", "rules_b.txt")

    assert detect_conflicts([a, b]) == []


def test_single_doc_returns_empty():
    from verification.conflict_detector import detect_conflicts

    a = _doc("Minimum attendance requirement is 75%.", "rules.txt")
    assert detect_conflicts([a]) == []


def test_unrelated_numbers_not_flagged():
    """Different numbers under genuinely different anchor contexts must not conflict."""
    from verification.conflict_detector import detect_conflicts

    a = _doc("Minimum attendance requirement is 75%.", "rules.txt")
    b = _doc("Scholarship disbursement is 10,000 rupees per semester.", "scholarship.txt")

    assert detect_conflicts([a, b]) == []


def test_currency_conflict_detected():
    from verification.conflict_detector import detect_conflicts

    a = _doc("Hostel fee is Rs. 25000 per semester.", "hostel_a.txt")
    b = _doc("Hostel fee is Rs. 30000 per semester.", "hostel_b.txt")

    conflicts = detect_conflicts([a, b])
    assert len(conflicts) == 1
    values = {v["value"] for v in conflicts[0]["values"]}
    assert "Rs. 25000" in values or "Rs.25000" in values or "25000" in " ".join(values)
    assert "Rs. 30000" in values or "Rs.30000" in values or "30000" in " ".join(values)

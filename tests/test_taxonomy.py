"""Task 2.2 — data/taxonomy.yaml schema."""

from __future__ import annotations

from pathlib import Path

import yaml


TAXONOMY_PATH = Path(__file__).resolve().parent.parent / "data" / "taxonomy.yaml"


REQUIRED_DOC_TYPES = {
    "handbook", "exam", "scholarship", "placement",
    "admission", "facilities", "other",
}

REQUIRED_DEPARTMENTS = {
    "CSE", "ECE", "EEE", "MECH", "CIVIL", "MBA", "MCA", "MSc",
}


def test_taxonomy_schema():
    assert TAXONOMY_PATH.exists(), f"data/taxonomy.yaml must exist at {TAXONOMY_PATH}"
    loaded = yaml.safe_load(TAXONOMY_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict), "taxonomy.yaml top-level must be a mapping"

    assert "doc_types" in loaded, "taxonomy.yaml missing 'doc_types'"
    assert "departments" in loaded, "taxonomy.yaml missing 'departments'"

    doc_types = loaded["doc_types"]
    departments = loaded["departments"]
    assert isinstance(doc_types, list) and all(isinstance(x, str) for x in doc_types)
    assert isinstance(departments, list) and all(isinstance(x, str) for x in departments)

    missing_dt = REQUIRED_DOC_TYPES - set(doc_types)
    missing_dept = REQUIRED_DEPARTMENTS - set(departments)
    assert not missing_dt, f"doc_types missing entries: {missing_dt}"
    assert not missing_dept, f"departments missing entries: {missing_dept}"

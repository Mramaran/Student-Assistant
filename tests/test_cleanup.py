"""Phase 0 cleanup tests for the UC4 pivot.

Asserts tutor-matching modules are gone from disk, no longer importable,
and the RAG-layer surface files (app.py, rag/, core/, services/) carry no
residual references to the deprecated tutor-matching names.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


TUTOR_MODULES = [
    "services.matching_engine",
    "services.competency_graph",
    "services.triage",
    "services.personalization",
    "services.classifier",
    "services.study_plan",
    "services.alerts",
    "decision.advisor",
    "decision.recommendation_engine",
    "decision.rules_engine",
    "decision.edge_handler",
]


TUTOR_TEST_PATHS = [
    "tests/test_advisor_extraction.py",
    "tests/test_advisor_logic.py",
    "tests/test_matching_quality.py",
    "tests/test_matching.py",
    "tests/test_scenarios.py",
    "tests/test_edge_cases.py",
    "tests/test_pipeline.py",
    "tests/eval_matching_offline.py",
    "tests/stress_test.py",
    "_test_intelligence.py",
    "_test_pipeline.py",
]


STRAY_PATHS = [
    "sample_data",
    "pages/1_\U0001f4ca_Analytics.py",
    "Dockerfile.api",
    "Dockerfile.streamlit",
    "docker-compose.yml",
    "PROJECT_FLOW.md",
    "SYSTEM_FLOW_DIAGRAM.md",
    "docs/TO_DELETE.md",
]


FORBIDDEN_REFERENCE_PATTERNS = [
    "matching_engine",
    "competency_graph",
    "recommendation_engine",
]


SCANNED_TARGETS = ["app.py", "rag", "core", "services"]


@pytest.mark.parametrize("mod_name", TUTOR_MODULES)
def test_no_tutor_modules_importable(mod_name):
    """Each deleted tutor-matching module must raise ModuleNotFoundError."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(mod_name)


def test_no_tutor_module_references():
    """No RAG-layer file may reference the deprecated tutor-matching names."""
    offenders = []
    for target in SCANNED_TARGETS:
        path = REPO_ROOT / target
        if not path.exists():
            continue
        files = [path] if path.is_file() else list(path.rglob("*.py"))
        for file in files:
            try:
                text = file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for pat in FORBIDDEN_REFERENCE_PATTERNS:
                if pat in text:
                    offenders.append(
                        f"{file.relative_to(REPO_ROOT).as_posix()} still mentions '{pat}'"
                    )
    assert offenders == [], "\n".join(offenders)


def test_tutor_tests_removed():
    """Tutor-matching test files and root-level test strays must not exist."""
    offenders = [p for p in TUTOR_TEST_PATHS if (REPO_ROOT / p).exists()]
    assert offenders == [], f"Still present: {offenders}"


def test_strays_removed():
    """Stray top-level artefacts must be deleted."""
    offenders = [p for p in STRAY_PATHS if (REPO_ROOT / p).exists()]
    assert offenders == [], f"Still present: {offenders}"


def test_requirements_aligned():
    """requirements.txt must reflect the Gemini + HuggingFace-only stack."""
    req_path = REPO_ROOT / "requirements.txt"
    assert req_path.exists(), "requirements.txt missing"
    text = req_path.read_text(encoding="utf-8", errors="ignore").lower()
    forbidden = ["langchain-ollama", "ollama", "groq", "networkx"]
    present_required = ["langchain-google-genai", "langchain-huggingface"]
    bad = [name for name in forbidden if name in text]
    missing = [name for name in present_required if name not in text]
    assert not bad, f"Forbidden dependencies still listed: {bad}"
    assert not missing, f"Required dependencies missing: {missing}"

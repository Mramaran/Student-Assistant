"""Task 8.2 / 8.3 / 8.4 — README and UC4 docs."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _read(path: Path) -> str:
    assert path.exists(), f"Expected {path} to exist"
    return path.read_text(encoding="utf-8")


# ── Task 8.2 — README ───────────────────────────────────────────────────────

def test_readme_is_uc4():
    text = _read(REPO_ROOT / "README.md")
    lower = text.lower()

    for phrase in ("use case 4", "rag student assistant", "langchain", "gemini", "streamlit"):
        assert phrase in lower, f"README must mention {phrase!r}"

    for cmd in ("pip install -r requirements.txt", "python build_db.py", "streamlit run app.py"):
        assert cmd in text, f"README must list quickstart command {cmd!r}"

    for forbidden in ("tutor", "matching", "competency"):
        # Allow substrings inside unrelated words only if the root is absent.
        # We assert the literal substring is absent, matching the plan spec.
        assert forbidden not in lower, f"README must not mention {forbidden!r}"


# ── Task 8.3 — UC4 pipeline doc ─────────────────────────────────────────────

def test_pipeline_doc_present():
    path = REPO_ROOT / "docs" / "architecture" / "UC4_PIPELINE.md"
    text = _read(path)
    for heading in ("## Pipeline", "## Modules", "## Sequence (single query)"):
        assert heading in text, f"{path.name} must contain heading {heading!r}"
    assert "->" in text, "UC4_PIPELINE.md must include ASCII arrows (`->`)"


# ── Task 8.4 — operator runbook ─────────────────────────────────────────────

_FENCED_BLOCK_RE = re.compile(r"```(?:bash|sh)\n.*?```", re.DOTALL)


def test_runbook_present():
    path = REPO_ROOT / "docs" / "RUN_UC4.md"
    text = _read(path)

    required_sections = ("## Prerequisites", "## Ingest corpus", "## Run app",
                         "## Run evaluation", "## Troubleshooting")
    for section in required_sections:
        assert section in text, f"Runbook must contain section {section!r}"

    # Each section must have at least one bash fenced block following it.
    positions = [text.index(s) for s in required_sections] + [len(text)]
    for i, section in enumerate(required_sections):
        segment = text[positions[i]:positions[i + 1]]
        assert _FENCED_BLOCK_RE.search(segment), (
            f"Section {section!r} must contain at least one ```bash fenced block"
        )

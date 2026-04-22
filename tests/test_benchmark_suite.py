"""Task 6.1 / 6.2 / 6.3 — benchmark suite shape, runner, regression gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import skip_on_quota


REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = REPO_ROOT / "eval" / "eval_benchmark.json"


VALID_CATEGORIES = {
    "handbook", "exam", "scholarship", "placement",
    "admission", "facilities", "redteam",
}

VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def test_benchmark_shape():
    items = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    assert isinstance(items, list)
    assert len(items) == 15, f"Benchmark must have exactly 15 items, got {len(items)}"

    ids = [item.get("id") for item in items]
    assert len(set(ids)) == 15, f"Duplicate ids found: {ids}"

    redteam_count = 0
    for item in items:
        required = {"id", "question", "gold_answer", "expected_sources", "category", "difficulty"}
        missing = required - set(item.keys())
        assert not missing, f"Item {item.get('id')} missing keys: {missing}"

        assert isinstance(item["expected_sources"], list) and item["expected_sources"], (
            f"Item {item['id']} must have a non-empty expected_sources list"
        )
        assert item["category"] in VALID_CATEGORIES, (
            f"Item {item['id']} has invalid category {item['category']!r}"
        )
        assert item["difficulty"] in VALID_DIFFICULTIES, (
            f"Item {item['id']} has invalid difficulty {item['difficulty']!r}"
        )

        if item["category"] == "redteam":
            redteam_count += 1

    assert redteam_count == 5, f"Must have exactly 5 redteam items, got {redteam_count}"


@skip_on_quota
def test_run_produces_results(tmp_path: Path):
    from eval.benchmark_suite import run_benchmark

    out = run_benchmark(subset="redteam", results_dir=tmp_path, max_items=2)
    assert "json_path" in out and "md_path" in out
    json_path = Path(out["json_path"])
    md_path = Path(out["md_path"])

    assert json_path.exists() and json_path.suffix == ".json"
    assert md_path.exists() and md_path.suffix == ".md"

    md_text = md_path.read_text(encoding="utf-8")
    for marker in ("recall@k", "citation_sources_valid", "redteam_block_rate"):
        assert marker in md_text, f"Markdown summary missing {marker!r}"

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "items" in data and isinstance(data["items"], list)
    assert data["items"], "Run must produce at least one item record"
    for record in data["items"]:
        required = {
            "id", "recall@k", "mrr", "answer_has_citation",
            "citation_sources_valid", "confidence", "verdict",
            "hallucination_flag", "pass",
        }
        missing = required - set(record.keys())
        assert not missing, f"Item record missing keys: {missing}"


def test_regression_gate(tmp_path: Path, monkeypatch):
    """Task 6.3 — baseline with recall@k=0.80 and current=0.70 must trigger regression."""
    from eval.benchmark_suite import run_benchmark, check_regression

    baseline = {
        "summary": {
            "recall@k":              0.80,
            "citation_sources_valid": 0.90,
        }
    }
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    current = {
        "summary": {
            "recall@k":              0.70,  # 10pp drop — beyond 5pp threshold
            "citation_sources_valid": 0.90,
        }
    }
    regressed, notes = check_regression(current, baseline, threshold_pp=5.0)
    assert regressed is True
    joined = "\n".join(notes)
    assert "REGRESSION" in joined and "recall@k" in joined and "10.0pp" in joined


def test_regression_gate_passes_when_within_threshold(tmp_path: Path):
    from eval.benchmark_suite import check_regression
    baseline = {"summary": {"recall@k": 0.80, "citation_sources_valid": 0.90}}
    current = {"summary": {"recall@k": 0.78, "citation_sources_valid": 0.92}}
    regressed, notes = check_regression(current, baseline, threshold_pp=5.0)
    assert regressed is False

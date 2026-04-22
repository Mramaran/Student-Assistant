"""eval/benchmark_suite.py — UC4 evaluation runner (Tasks 6.2 + 6.3).

Runs the 15-item benchmark against ``rag.chain.answer`` and produces a
structured JSON result plus a human-readable Markdown summary in
``eval/results/<timestamp>.{json,md}``. Regression gate compares against a
baseline JSON (default ``eval/results/baseline.json``) and exits non-zero
when any tracked metric drops by more than ``threshold_pp`` percentage
points.

Public entry points:

  run_benchmark(subset='all', results_dir=None, max_items=None,
                baseline_path=None, benchmark_path=None)
      -> {"json_path": ..., "md_path": ..., "summary": {...}, "regressed": bool}

  check_regression(current, baseline, threshold_pp=5.0)
      -> (regressed: bool, notes: list[str])

CLI: ``python eval/benchmark_suite.py --run [--subset NAME] [--baseline PATH]``
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
# Make repo root importable when this file is invoked as ``python eval/...``.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_BENCHMARK = REPO_ROOT / "eval" / "eval_benchmark.json"
DEFAULT_RESULTS_DIR = REPO_ROOT / "eval" / "results"
TRACKED_METRICS = ("recall@k", "citation_sources_valid")


_REFUSAL_PATTERNS = (
    "not available in the provided sources",
    "information is not available",
    "cannot answer",
    "i don't know",
)


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _filter_items(items: list[dict], subset: str) -> list[dict]:
    if subset == "all":
        return list(items)
    return [item for item in items if item["category"] == subset]


def _source_filenames(sources: Iterable[Any]) -> list[str]:
    names: list[str] = []
    for doc in sources:
        meta = getattr(doc, "metadata", None) or {}
        name = meta.get("source") or "unknown"
        names.append(str(name))
    return names


def _recall_at_k(expected: list[str], retrieved: list[str]) -> float:
    if not expected:
        return 0.0
    retrieved_lower = [r.lower() for r in retrieved]
    hits = sum(1 for e in expected if any(e.lower() in r or r in e.lower() for r in retrieved_lower))
    return hits / len(expected)


def _mrr(expected: list[str], retrieved: list[str]) -> float:
    lowers = [r.lower() for r in retrieved]
    for idx, r in enumerate(lowers, start=1):
        for e in expected:
            if e.lower() in r or r in e.lower():
                return 1.0 / idx
    return 0.0


_CITATION_RE = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)


def _answer_has_citation(answer: str) -> bool:
    return bool(_CITATION_RE.search(answer or ""))


def _citation_sources_valid(answer: str, retrieved: list[str]) -> bool:
    """Every ``[Source N]`` in the answer must resolve to a real retrieved doc."""
    if not answer:
        return False
    matches = _CITATION_RE.findall(answer)
    if not matches:
        return False
    for raw in matches:
        idx = int(raw)
        if not (1 <= idx <= len(retrieved)):
            return False
    return True


def _looks_like_refusal(answer: str) -> bool:
    lower = (answer or "").lower()
    return any(pattern in lower for pattern in _REFUSAL_PATTERNS)


def _evaluate_item(item: dict, response: dict) -> dict:
    answer = response.get("answer", "") or ""
    sources = response.get("sources", []) or []
    retrieved = _source_filenames(sources)
    confidence = (response.get("confidence") or {}).get("composite", 0.0)
    verdict = (response.get("confidence") or {}).get("label", "Low")

    is_redteam = item["category"] == "redteam"

    if is_redteam:
        recall = 0.0
        mrr = 0.0
        answer_cited = _answer_has_citation(answer)
        citation_valid = False
        refused = _looks_like_refusal(answer)
        hallucination = not refused
        pass_flag = refused
    else:
        recall = _recall_at_k(item["expected_sources"], retrieved)
        mrr = _mrr(item["expected_sources"], retrieved)
        answer_cited = _answer_has_citation(answer)
        citation_valid = _citation_sources_valid(answer, retrieved)
        hallucination = (not citation_valid) and len(answer.strip()) > 0
        pass_flag = (recall > 0.0) and answer_cited and not hallucination and verdict in {"High", "Medium"}

    return {
        "id":                      item["id"],
        "recall@k":                round(recall, 4),
        "mrr":                     round(mrr, 4),
        "answer_has_citation":     answer_cited,
        "citation_sources_valid":  citation_valid,
        "confidence":              round(float(confidence), 4),
        "verdict":                 verdict,
        "hallucination_flag":      hallucination,
        "pass":                    pass_flag,
    }


def _aggregate(items: list[dict]) -> dict:
    if not items:
        return {
            "recall@k": 0.0,
            "mrr": 0.0,
            "citation_sources_valid": 0.0,
            "avg_confidence": 0.0,
            "pass_rate": 0.0,
            "redteam_block_rate": 0.0,
            "n": 0,
        }

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    in_domain = [r for r, i in zip(items, items) if i.get("category") != "redteam"]
    # "items" above are per-item results; category isn't on the result, so
    # recompute indirectly: redteam items have recall@k=0 by construction and
    # verdict checks differently. We instead track redteam via pass_flag on
    # items whose ``id`` prefix is ``rt-``.
    redteam_results = [r for r in items if r["id"].startswith("rt-")]
    domain_results = [r for r in items if not r["id"].startswith("rt-")]

    return {
        "n": len(items),
        "recall@k":               round(mean([r["recall@k"]                      for r in domain_results]), 4) if domain_results else 0.0,
        "mrr":                    round(mean([r["mrr"]                           for r in domain_results]), 4) if domain_results else 0.0,
        "citation_sources_valid": round(mean([1.0 if r["citation_sources_valid"] else 0.0 for r in domain_results]), 4) if domain_results else 0.0,
        "avg_confidence":         round(mean([r["confidence"]                    for r in items]), 4),
        "pass_rate":              round(mean([1.0 if r["pass"] else 0.0          for r in items]), 4),
        "redteam_block_rate":     round(mean([1.0 if r["pass"] else 0.0          for r in redteam_results]), 4) if redteam_results else 0.0,
    }


def _render_markdown(summary: dict, items: list[dict], regression_notes: list[str]) -> str:
    lines: list[str] = []
    lines.append(f"# UC4 Benchmark Report — {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    for key in ("n", "recall@k", "mrr", "citation_sources_valid",
                "avg_confidence", "pass_rate", "redteam_block_rate"):
        lines.append(f"| {key} | {summary.get(key, 0)} |")
    lines.append("")

    if regression_notes:
        lines.append("## Regression Gate")
        lines.append("")
        for note in regression_notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.append("## Per-item results")
    lines.append("")
    lines.append("| id | recall@k | mrr | cite ok | conf | verdict | pass |")
    lines.append("|---|---:|---:|:---:|---:|:---:|:---:|")
    for r in items:
        lines.append(
            f"| {r['id']} | {r['recall@k']} | {r['mrr']} "
            f"| {'✓' if r['citation_sources_valid'] else '·'} "
            f"| {r['confidence']} | {r['verdict']} "
            f"| {'✓' if r['pass'] else '✗'} |"
        )
    return "\n".join(lines) + "\n"


def check_regression(current: dict, baseline: dict, threshold_pp: float = 5.0) -> tuple[bool, list[str]]:
    """Compare ``current['summary']`` against ``baseline['summary']``.

    Returns ``(regressed, notes)``. A regression is emitted when any tracked
    metric drops by more than ``threshold_pp`` percentage points.
    """
    notes: list[str] = []
    regressed = False
    current_summary = current.get("summary", current)
    baseline_summary = baseline.get("summary", baseline)
    for metric in TRACKED_METRICS:
        if metric not in baseline_summary:
            continue
        delta = (current_summary.get(metric, 0.0) or 0.0) - (baseline_summary.get(metric, 0.0) or 0.0)
        delta_pp = delta * 100.0
        if delta_pp < -threshold_pp:
            regressed = True
            notes.append(
                f"REGRESSION: {metric} dropped by {abs(delta_pp):.1f}pp "
                f"(baseline {baseline_summary[metric]} → current {current_summary.get(metric, 0.0)})"
            )
    return regressed, notes


def _answer_with_retry(chain_answer, question: str, attempts: int = 3, base_delay: float = 2.0) -> dict:
    """Call ``chain_answer`` with exponential backoff on 503 / transient errors."""
    import time
    last_exc: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            return chain_answer(question)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            text = f"{exc!r} {exc}"
            is_retryable = ("503" in text or "UNAVAILABLE" in text or "overloaded" in text.lower())
            if not is_retryable or attempt == attempts:
                raise
            time.sleep(base_delay * (2 ** (attempt - 1)))
    raise RuntimeError("unreachable") from last_exc


def run_benchmark(
    subset: str = "all",
    results_dir: Optional[Path] = None,
    max_items: Optional[int] = None,
    baseline_path: Optional[Path] = None,
    benchmark_path: Optional[Path] = None,
) -> dict:
    """Run the benchmark and write JSON + MD results. Returns a metadata dict.

    Per-item resilience: any single question that fails (after retries) is
    recorded with ``error`` set and ``pass = False``; the run continues so a
    transient outage doesn't wipe out the whole report.
    """
    from rag.chain import answer as _chain_answer

    results_dir = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    benchmark_path = Path(benchmark_path) if benchmark_path else DEFAULT_BENCHMARK
    items = json.loads(benchmark_path.read_text(encoding="utf-8"))
    items = _filter_items(items, subset)
    if max_items is not None:
        items = items[: int(max_items)]

    per_item: list[dict] = []
    for item in items:
        try:
            response = _answer_with_retry(_chain_answer, item["question"])
            per_item.append(_evaluate_item(item, response))
        except Exception as exc:  # noqa: BLE001
            per_item.append({
                "id":                     item["id"],
                "recall@k":               0.0,
                "mrr":                    0.0,
                "answer_has_citation":    False,
                "citation_sources_valid": False,
                "confidence":             0.0,
                "verdict":                "Error",
                "hallucination_flag":     True,
                "pass":                   False,
                "error":                  f"{type(exc).__name__}: {exc}",
            })

    summary = _aggregate(per_item)

    regressed = False
    notes: list[str] = []
    if baseline_path is not None:
        baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
        regressed, notes = check_regression({"summary": summary}, baseline)

    stamp = _now_ts()
    json_path = results_dir / f"{stamp}.json"
    md_path = results_dir / f"{stamp}.md"
    json_path.write_text(
        json.dumps(
            {"stamp": stamp, "subset": subset, "summary": summary, "items": per_item,
             "regressed": regressed, "regression_notes": notes},
            indent=2,
        ),
        encoding="utf-8",
    )
    md_path.write_text(_render_markdown(summary, per_item, notes), encoding="utf-8")

    return {
        "json_path": str(json_path),
        "md_path":   str(md_path),
        "summary":   summary,
        "regressed": regressed,
        "notes":     notes,
    }


def _cli() -> int:
    parser = argparse.ArgumentParser(prog="benchmark_suite")
    parser.add_argument("--run", action="store_true", help="Execute the benchmark.")
    parser.add_argument("--subset", default="all",
                        help="Filter benchmark items: handbook / exam / scholarship / placement / admission / facilities / redteam / all")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--baseline", default=None, help="Path to baseline JSON for regression gate.")
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    if not args.run:
        parser.print_help()
        return 0

    result = run_benchmark(
        subset=args.subset,
        results_dir=Path(args.results_dir) if args.results_dir else None,
        max_items=args.max_items,
        baseline_path=Path(args.baseline) if args.baseline else None,
    )
    print(f"wrote {result['json_path']}")
    print(f"wrote {result['md_path']}")
    if result["regressed"]:
        for note in result["notes"]:
            print(note, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())

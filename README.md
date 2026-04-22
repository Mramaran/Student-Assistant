# UC4 RAG Student Assistant

End-to-end **Use Case 4** implementation: an academic-domain RAG Student
Assistant that answers questions grounded in the institute's documents,
cites every claim, scores its own confidence, surfaces contradictions, and
escalates to a human when it can't answer safely.

Built with **LangChain** (LCEL composition), **Gemini** as the generator,
HuggingFace sentence-transformer embeddings, ChromaDB, and **Streamlit** for
the front-end.

---

## Quickstart

```bash
pip install -r requirements.txt
python build_db.py
streamlit run app.py
```

Before the first run, create a `.env` in the repo root with:

```
GEMINI_API_KEY=...            # Google AI Studio key
GEMINI_MODEL=gemini-2.5-flash-lite   # optional; defaults to gemini-2.5-flash
ESCALATION_EMAIL=faculty@example.edu # who receives escalation mailto links
```

The corpus lives under `data/`. Drop `.txt`, `.pdf`, `.docx`, `.md`, or image
files there before running `python build_db.py`. Red-team / adversarial docs
belong in `data/redteam/` and are excluded from ingestion by default.

---

## Features (5 improvements on top of the base pipeline)

1. **Confidence scoring** — every answer returns a 4-factor composite
   (cosine, evidence alignment, corroboration, availability) with fixed
   weights and a `High / Medium / Low` label rendered in the UI.
2. **Contradiction alerts** — a rule-based anchor detector scans the
   retrieved chunks for conflicting numeric facts (percent, Rs/USD, dates)
   and surfaces a banner + side-by-side source comparison.
3. **Human escalation** — `Low`-confidence answers auto-create a ticket in
   `feedback/escalations.jsonl` and render a pre-filled `mailto:` button.
   An "Escalate anyway" button is available at every confidence level.
4. **Evaluation benchmark suite** — 15-item JSON benchmark (10 in-domain +
   5 red-team) run by `python eval/benchmark_suite.py --run`. Produces
   JSON + Markdown reports and enforces a 5pp regression gate against
   `eval/results/baseline.json`.
5. **Safety guardrails** — query-in prompt-injection filter, ingest-time
   `risk_flag` on suspicious chunks, and outbound PII scrub on every
   generated answer.

---

## Layout

```
core/         config, trace writer
data/         corpus (gitignored manifest.json)
embeddings/   HuggingFace embedder + Chroma store
eval/         benchmark JSON, runner, results
feedback/     trace + escalations JSONL logs, analytics KPIs
ingestion/    loader, cleaner, chunker
rag/          chain (LCEL), retriever, generator shim
services/     escalation, safety_handler
tests/        pytest suite (conftest forces gemini-2.5-flash-lite)
verification/ confidence, conflict_detector, verifier
```

---

## Running the evaluation

```bash
python eval/benchmark_suite.py --run --subset redteam
python eval/benchmark_suite.py --run --subset all \
    --baseline eval/results/baseline.json
```

The runner writes `eval/results/<timestamp>.json` and `.md`. The Markdown
includes per-item rows and the run summary. CI can fail on regression by
checking the exit code.

---

## Troubleshooting

- **`RESOURCE_EXHAUSTED` (429)** — free-tier Gemini gives 20 requests/day
  per project per model. Rotate keys, switch `GEMINI_MODEL` to a different
  variant, or wait for the daily reset. `tests/conftest.py` already pins
  the test-run model to `gemini-2.5-flash-lite`.
- **Empty Chroma DB** — re-run `python build_db.py`. The `data/manifest.json`
  is gitignored and rebuilds incrementally using sha1 per file.
- **Streamlit module errors** — confirm `streamlit` is on the active Python
  environment and `pip install -r requirements.txt` completed.

---

## Documentation

- [`docs/architecture/UC4_PIPELINE.md`](docs/architecture/UC4_PIPELINE.md) — pipeline diagram + per-module responsibilities.
- [`docs/RUN_UC4.md`](docs/RUN_UC4.md) — operator runbook (prereqs, ingest, run, eval, troubleshoot).
- [`docs/plans/2026-04-21-uc4-rag-student-assistant-plan.md`](docs/plans/2026-04-21-uc4-rag-student-assistant-plan.md) — the implementation plan this repo follows.
- [`docs/commits/uc4-commits.md`](docs/commits/uc4-commits.md) — deferred git commands for the pivot (never applied locally yet).

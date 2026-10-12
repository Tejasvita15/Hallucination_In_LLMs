# One-day finish: analysis and report assets

Do not rerun the full 99,876-output suite before submission. Its predictions are already audited. Generation cannot supply missing human labels. Formatting changes would create a new protocol and cannot silently replace the frozen results.

## Run this from the repository root

```text
python scripts/build_thesis_assets.py
```

On this Windows workspace the plotting dependencies are installed in `.report-deps`, which the script discovers automatically. On another machine, install the small reporting environment first:

```bash
python -m venv .venv-report
source .venv-report/bin/activate
python -m pip install -r requirements-report.txt
python scripts/build_thesis_assets.py
```

This is a CPU task, so no H100 or PBS submission is necessary. It reads the four local audit JSON files specified in the script. Those audit directories must be copied to the cluster too if running there; some are currently untracked locally. Running locally is the simplest option.

## Outputs

`assets/figures/`: vector PDF/SVG and 250-dpi PNG plots.

`assets/tables/`: generated LaTeX and Markdown tables. `assets/data/` stores their displayed values as JSON. The original full-precision audit JSON remains authoritative.

`assets/manifest.json`: hashes of the input audits and generator, plotting version and captions. `assets/figure_includes.tex`: optional figure snippets using `graphicx`. In Overleaf, upload the assets directory beside the thesis source, then insert selected snippets in the relevant chapters. Do not blindly append every figure and duplicate every existing table.

Figures cover official TruthfulQA, strict/recovered prompted-choice outcomes, paired improvement intervals, HaluEval/RAGTruth format failures, English and multilingual Mu-SHROOM, RAGTruth IoU sensitivity, exploratory EN/ZH results and descriptive token probabilities. No plot labels raw token likelihood as factuality probability. No new interval is invented from summary data.

## Improvements implemented now

- One reporting command, using audited machine-readable results instead of stale manually entered plot values.
- Consistent model ordering, readable legends, vector exports and explicit metric scales.
- Strict and post-hoc recovery shown separately.
- Count and audit-status validation before plotting; input hashes recorded.
- Separate generation-invalidity plots from factuality claims.
- Existing human-review import rejects incomplete packets, duplicate IDs and altered source answers.

## Today's research priority

Complete `english_review.html` and export the 90 labels. Once complete, use the existing analysis script:

```text
python report/submission_20261002/analyze_human_review.py PATH_TO_COMPLETED_CSV --reviewer "Tejasvita Jain" --review-date 2026-10-13 --out report/submission_20261002/human_analysis_20261013
```

The output directory must be new to preserve annotation history. This produces confidence summaries but does not automatically alter thesis claims. Return the CSV for review and integration. Do not use AI labels to bypass this validation. The packet covers baseline English answers, not new mitigation generations.

## Defer until after submission

DeepSeek-specific structured-output prompts, a revised multilingual explanation validator, checkpoint-write improvements and newly labelled CAD/evidence-first comparisons are valuable future work. Keep them versioned separately. The supplied empty checkpoint files have an unknown cause; do not declare a checkpoint writer bug or delete cluster originals without investigation.

Human review, factuality verification and PDF proofreading remain necessary. No code change can guarantee a stronger or positive research result within one day.

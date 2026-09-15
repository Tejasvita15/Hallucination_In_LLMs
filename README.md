# Hallucinations in LLMs

Structured research code for benchmarking hallucination, token-level confidence,
and cross-lingual refusal calibration in open-weight LLMs.

## Project Layout

```text
configs/
  benchmark_suite.full.json     Four full hallucination benchmarks plus factuality control
  benchmark_suite_protocol.json Frozen benchmark roles, splits, and primary metrics
  controlled_benchmark.smoke.json
                               20-pair controlled-negative generation pilot
  controlled_benchmark.full.json
                               Reviewed 500-pair controlled benchmark protocol
  controlled_confirmatory_protocol.json
                               Frozen confirmatory hypotheses and statistics
  detection.controlled.confirmatory.json
                               Qwen2.5 English/Chinese confirmatory run
  detection.halueval-qa.full.json
                               Complete official HaluEval-QA split in English
  detection_analysis.controlled.json
                               Frozen-split confirmatory statistics
  detection.example.json       Selected NLLB-600M reviewed smoke test
  detection.nllb-600m.json     Explicit NLLB-600M configuration
  detection.nllb-600m.full.json
                               Strict 150-sample full-study configuration
  detection.nllb-1.3b.json     Optional matched NLLB-1.3B ablation
  detection_analysis.example.json
                               Held-out calibration and statistics protocol
  halueval_reviews.json        Versioned translation and label review decisions
  experiment.example.json      Example experiment configuration
  models.example.json          Open-weight model list
  languages.example.json       Language list and prompt templates
jobs/
  submit_controlled_benchmark_tc2.sh
                               A100 job for controlled-negative generation
  submit_controlled_detection_tc2.sh
                               Confirmatory controlled benchmark evaluation
  submit_detection_tc2.sh      HaluEval detection job for CCDS MGPU-TC2
  submit_external_benchmark_tc2.sh
                               RAGTruth, Mu-SHROOM, or TruthfulQA full run
  submit_hallulens_official_tc2.sh
                               One official HalluLens task per Slurm job
  submit_halueval_full_tc2.sh   24-hour full HaluEval-QA job
  submit_nllb_ablation_tc2.sh  Two-task NLLB model-size ablation
  submit_experiment_ntu.sh     Slurm entrypoint for NTU GPU cluster
scripts/
  prepare_benchmark_suite.py   Downloads, normalizes, hashes, and manifests benchmarks
  run_external_benchmark.py    RAGTruth, Mu-SHROOM, and TruthfulQA evaluator
  record_hallulens_results.py  Hashes official HalluLens task outputs
  aggregate_benchmark_suite.py Completeness gate and benchmark-level tables
  build_length_controlled_halueval.py
                               Generates and freezes reviewed, length-controlled negatives
  analyze_detection_results.py Held-out calibration and statistical analysis
  prefetch_hf_assets.py        Downloads configured models and datasets
  run_halueval_detection.py    Paired multilingual hallucination detector
  run_crosslingual.py          Main cross-lingual experiment runner
  run_hallulens_task.py        HalluLens-style task runner
  aggregate_structured.py      Aggregates structured JSONL outputs
src/hallu_llm/
  benchmark_suite.py           External schemas, span parsing, and suite metrics
  data.py                      Dataset loading and sample normalization
  detection.py                 Detection instances, scoring, and metrics
  detection_analysis.py        Calibration, bootstrap, and length controls
  evaluation.py                Core scoring loops
  metrics.py                   Metrics and confidence helpers
  models.py                    HF model loading and generation
  prompts.py                   Prompt templates
  text.py                      Conservative sentence-boundary repair
  translation.py               Translation helpers
  translation_quality.py       Translation diagnostics and quality gate
  utils.py                     IO, seeds, filesystem helpers
```

The original single-file scripts are kept in the repository for continuity.
Use the `scripts/` entrypoints for new experiments.

## Full Benchmark Suite

### Enabled production models

The full suite, full HaluEval-QA, controlled confirmatory detector, and 150-pair
translation-study configurations enable these three evaluation checkpoints:

| Configured name | Hugging Face checkpoint |
| --- | --- |
| `qwen2.5-7b` | `Qwen/Qwen2.5-7B-Instruct` |
| `llama3.1-8b` | `meta-llama/Llama-3.1-8B-Instruct` |
| `deepseek-r1-7b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` |

The paired HaluEval runner iterates over all configured models. External
benchmark jobs run one model per submission; select either added model with
`MODEL_NAME`, for example:

```bash
sbatch --export=ALL,BENCHMARK=ragtruth,MODEL_NAME=llama3.1-8b jobs/submit_external_benchmark_tc2.sh
sbatch --export=ALL,BENCHMARK=ragtruth,MODEL_NAME=deepseek-r1-7b jobs/submit_external_benchmark_tc2.sh
```

Repeat for `mu_shroom` and `truthfulqa`. Prefetch the full suite config to cache
all three checkpoints; Llama requires an account with approved model access.
Enabling a checkpoint does not download weights or submit GPU jobs.

This model-list expansion adds replication comparisons; the frozen primary
Qwen hypothesis and Qwen proposal generator remain as recorded. Existing
experiment directory names are retained for compatibility. DeepSeek's 7B
checkpoint is the Qwen-based R1 distillation. The existing four-token binary
decision protocol is a constrained-output evaluation, not a full reasoning
evaluation; inspect its generated decisions and invalid-output rate in a smoke
run before full inference.

The official HalluLens endpoint must be configured separately for each model.
The current suite aggregator selects the first configured model (Qwen); it
does not yet produce a three-model comparison table. Per-model prediction and
summary files from the evaluation runners remain separate.

The production suite uses four core hallucination benchmarks and one separate
factuality control. These are not interchangeable tasks, so their metrics are
reported in separate rows rather than averaged into a synthetic overall score.

| Benchmark | Full evaluation scope | Role | Upstream license |
| --- | --- | --- | --- |
| HaluEval-QA | Every official QA pair, English | Paired grounded/hallucinated response detection | MIT |
| HalluLens | Every prompt from all four official task scripts | Dynamic hallucination and refusal generation | Mostly CC-BY-NC; see upstream exceptions |
| RAGTruth | Every good-quality test response | RAG response and hallucinated-span detection | MIT |
| Mu-SHROOM | Every test response in all 14 languages | Primary cross-lingual span benchmark | CC-BY-4.0 |
| TruthfulQA | Every official question | Separate factuality control, not a hallucination score | Apache-2.0 |

The frozen decisions are in `configs/benchmark_suite_protocol.json`. Mu-SHROOM
validation is downloaded and retained for development, but it is never pooled
with the final test metrics. The complete HaluEval-QA benchmark is English-only;
the reviewed controlled HaluEval experiment remains the focused English/Chinese
study, and full Mu-SHROOM supplies broad cross-lingual coverage.

On the NTU login node, download the Hugging Face model and datasets, clone the
official repositories, normalize the static benchmarks, and record source
hashes. HalluLens data is a separate, large download and is explicitly enabled:

```bash
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
python scripts/prefetch_hf_assets.py \
  --config configs/benchmark_suite.full.json
python scripts/prepare_benchmark_suite.py \
  --config configs/benchmark_suite.full.json \
  --download \
  --download-hallulens-data
```

Check `data/benchmarks/prepared/manifest.json` before GPU submission. It records
repository commits, file hashes, row counts, exclusions, split counts, and the
exact suite-config hash. Raw and normalized datasets are intentionally ignored
by Git; the provenance manifest and official source URLs make them reproducible
without committing redistributed benchmark data.

Preparation can also be repeated for one source with `--benchmark NAME`; when
the config hash is unchanged, its new record is merged into the existing suite
manifest rather than deleting the other prepared-source records.

First run small smoke jobs. Supplying `MAX_SAMPLES` marks the result as partial:

```bash
sbatch --export=ALL,BENCHMARK=ragtruth,MAX_SAMPLES=20 \
  jobs/submit_external_benchmark_tc2.sh
sbatch --export=ALL,BENCHMARK=mu_shroom,MAX_SAMPLES=20 \
  jobs/submit_external_benchmark_tc2.sh
sbatch --export=ALL,BENCHMARK=truthfulqa,MAX_SAMPLES=20 \
  jobs/submit_external_benchmark_tc2.sh
```

After inspecting the smoke outputs, submit full runs by omitting
`MAX_SAMPLES`. Each benchmark is a separate job, not a Slurm array, and can
resume its compatible `predictions.jsonl` checkpoint after interruption:

```bash
sbatch --export=ALL,BENCHMARK=ragtruth \
  jobs/submit_external_benchmark_tc2.sh
sbatch --export=ALL,BENCHMARK=mu_shroom \
  jobs/submit_external_benchmark_tc2.sh
sbatch --export=ALL,BENCHMARK=truthfulqa \
  jobs/submit_external_benchmark_tc2.sh
sbatch jobs/submit_halueval_full_tc2.sh
```

HalluLens is dynamic rather than a static JSONL test set. Use its official
checkout and configure `data/benchmarks/raw/HalluLens/utils/lm.py` for the model
endpoint as required by the upstream project. Submit each official task
separately, for example:

```bash
sbatch --export=ALL,TASK=precisewikiqa \
  jobs/submit_hallulens_official_tc2.sh
sbatch --export=ALL,TASK=longwiki \
  jobs/submit_hallulens_official_tc2.sh
sbatch --export=ALL,TASK=mixedentities \
  jobs/submit_hallulens_official_tc2.sh
sbatch --export=ALL,TASK=generatedentities \
  jobs/submit_hallulens_official_tc2.sh
```

The generated-entities task additionally requires the upstream search and judge
credentials. After all four official tasks finish, hash their final result
files into the suite manifest:

```bash
python scripts/record_hallulens_results.py --full-split \
  --task-result 'PreciseWikiQA=<result-file>' \
  --task-result 'LongWiki=<result-file>' \
  --task-result 'NonExistentRefusal/MixedEntities=<result-file>' \
  --task-result 'NonExistentRefusal/GeneratedEntities=<result-file>'
```

Finally, create the paper-facing table and enforce that all four core
benchmarks are complete:

```bash
python scripts/aggregate_benchmark_suite.py
```

Use `--allow-incomplete` only while monitoring progress. RAGTruth reports
response-level classification, calibration, and hard-span metrics. Mu-SHROOM
reports micro results plus per-language and macro-language metrics. The runner
stores every prompt result and checkpoints every 25 samples for auditability.

## Quick Start

Install dependencies in your cluster environment:

```bash
python -m pip install torch transformers accelerate datasets sentencepiece protobuf safetensors matplotlib pandas
```

Run the 20-source-sample HaluEval detection smoke test:

```bash
python scripts/run_halueval_detection.py --config configs/detection.example.json
```

## Length-Controlled Confirmatory Workflow

The next project phase replaces HaluEval's easily detected long negatives with
human-approved candidates matched to each grounded answer. Generation does not
automatically certify a candidate: every proposal remains pending until a
reviewer approves or corrects it with a written reason.

The prospective analysis commitments are recorded in
`configs/controlled_confirmatory_protocol.json`, dated September 1, 2026.
Commit that file together with the benchmark, detector, and analysis configs
before running the confirmatory detector.

Prefetch the source dataset, proposal generator, detector, and NLLB translator
while Hugging Face network access is enabled:

```bash
python scripts/prefetch_hf_assets.py \
  --config configs/controlled_benchmark.full.json \
  --config configs/detection.controlled.confirmatory.json
```

Generate 550 proposals on one NTU A100. The job resumes compatible proposal
caches automatically and does not use a Slurm array:

```bash
sbatch jobs/submit_controlled_benchmark_tc2.sh
```

Two independent reviewers separately edit
`proposal_review_reviewer_1.csv` and `proposal_review_reviewer_2.csv` under
`results/benchmark/length_controlled_halueval_500_v1/`. Reviewers should not
inspect each other's decisions before both files are complete. For every row,
set `review_status` to `approved` or `rejected`, add a non-empty
`review_reason`, and record a distinct, non-empty `reviewer`. Use
`reviewed_candidate` only when the generated proposal needs correction.
Approved candidates are revalidated against the 25% length and sentence-count
constraints, answer style, and named-entity count.

Finalize the reviewed dataset on a login or CPU node. This writes the paired
JSONL dataset and freezes the source-level development/test split before any
detector results are inspected:

```bash
python scripts/build_length_controlled_halueval.py \
  --config configs/controlled_benchmark.full.json \
  --finalize-only
```

If the two decisions or approved candidates disagree, this command writes
`proposal_adjudication.csv` and stops at the review gate. A third reviewer
completes the adjudication rows, after which the same finalization command is
run again. The quality report records percent agreement, Cohen's kappa, and
the number of adjudicated pairs.

Prepare the audited Chinese translation and run the Qwen2.5 baseline:

```bash
sbatch jobs/submit_controlled_detection_tc2.sh
```

If the strict Chinese translation gate stops the job, inspect its quality JSON
and audit CSV, record corrections in `configs/controlled_halueval_reviews.json`,
and resubmit. Do not bypass the gate for the confirmatory run.

Analyze the completed run with the already-frozen split:

```bash
python scripts/analyze_detection_results.py \
  --config configs/detection_analysis.controlled.json
```

The final controlled dataset is released only when at least 500 proposals are
approved and every proposal has two completed independent reviews plus any
required adjudication. The length-only baseline must
remain in the final tables; the controlled construction is successful only if
that shortcut falls substantially toward chance.

The primary example uses `facebook/nllb-200-distilled-600M` for
English-to-Simplified-Chinese translation with the NLLB language codes
`eng_Latn` and `zho_Hans`. Four-beam deterministic decoding is used. Knowledge
passages are split into sentences before translation, including repair of
missing HaluEval boundaries such as `Thun.FC Thun`, so NLLB cannot silently
discard all later evidence sentences in one generation.

Language-neutral numeric answers are preserved without translation. A question
that loses its interrogative intent or a source number is retried as independent
clauses, and the retry is accepted only when it reduces those errors.
Numeric answer alignment uses extracted number values, so values embedded next
to Chinese characters, such as `1946年`, are matched without accepting partial
matches such as `16` inside `2016`.

Reviewed translation corrections and accepted semantic alignments are stored in
`configs/halueval_reviews.json`. The original NLLB text remains in the dataset
cache and audit CSV. Changing the review manifest changes its recorded hash and
refreshes the reviewed cache rows without rerunning NLLB when the source and
translator settings are unchanged. Exact accepted warnings are retained in the
automatic-warning audit but removed from the unresolved gate only with a
sample-specific reason. The same manifest records source pairs whose alleged
hallucinated answer is actually supported by the evidence. Predictions are
still generated and saved for those pairs, but they are excluded from primary
metrics with the review reason attached.

The runner evaluates both the grounded and hallucinated candidate for every
source sample. The smoke test therefore produces 40 balanced decisions per
language. It asks for a strict `YES` (hallucinated) or `NO` (grounded) response
and uses that generated decision for classification. Normalized first-token
probability over those labels is retained as the confidence score.

The paired runner requires HaluEval subset `qa`. Do not change it to
`qa_samples`: that subset contains one candidate plus a `yes`/`no` label rather
than separate `right_answer` and `hallucinated_answer` fields. The loader rejects
that incompatible schema before model inference.

Prefetch the selected model and dataset from a login node before submitting the
offline compute job:

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
  python scripts/prefetch_hf_assets.py \
    --config configs/detection.nllb-600m.json
```

Run the selected NLLB-600M retry smoke test on the NTU CCDS TC2 partition:

```bash
sbatch --export=ALL,CONFIG=configs/detection.nllb-600m.json \
  jobs/submit_detection_tc2.sh
```

`jobs/submit_nllb_ablation_tc2.sh` remains available when a matched
600M-versus-1.3B comparison is needed. Its two tasks use the same seed, source
pairs, detector, prompts, and decoding settings.

Run the strict 150-sample translation preflight first. It creates the complete
dataset audit without loading Qwen and aborts if the quality gate fails:

```bash
sbatch --export=ALL,CONFIG=configs/detection.nllb-600m.full.json,PREPARE_ONLY=1 \
  jobs/submit_detection_tc2.sh
```

After `datasets/zh.quality.json` reports `"quality_gate_passed": true`, submit
the detector run with the same config. The validated translation cache is reused:

```bash
sbatch --export=ALL,CONFIG=configs/detection.nllb-600m.full.json \
  jobs/submit_detection_tc2.sh
```

To evaluate every valid row in the HaluEval split instead of the planned
150-sample study, set `ALL_SAMPLES=1` and give the experiment a distinct name.
This can require substantially more cluster time:

```bash
sbatch --export=ALL,CONFIG=configs/detection.nllb-600m.full.json,ALL_SAMPLES=1,EXPERIMENT_NAME=crosslingual_halueval_detection_nllb_600m_all_v4 \
  jobs/submit_detection_tc2.sh
```

Run the focused unit tests:

```bash
python -m unittest discover -s tests -v
```

## Held-out Statistical Analysis

The detector probabilities can be analyzed without loading Qwen, NLLB, or
PyTorch. The analysis assigns complete source pairs to a deterministic
development/test split, fits every threshold or probability calibrator on the
development split only, and reports test-only results:

```bash
python scripts/analyze_detection_results.py \
  --results-dir results/detection/crosslingual_halueval_detection_nllb_600m_full_150_v4
```

The equivalent tracked configuration is:

```bash
python scripts/analyze_detection_results.py \
  --config configs/detection_analysis.example.json
```

The default protocol uses 30% of source pairs for development, 70% for test,
2,000 source-pair bootstrap replicates, English as the reference language, and
a fixed seed of 42. It compares:

- the saved generated `YES`/`NO` decision
- the uncalibrated decision-token probability at threshold 0.5
- thresholds fitted on pooled languages, English, or each target language
- language-specific temperature scaling
- pooled, English-transfer, and language-specific Platt scaling
- a candidate-length-only logistic baseline
- a model-plus-length logistic diagnostic

Temperature scaling changes probability calibration but cannot move a 0.5
decision boundary. Platt scaling can correct both scale and intercept drift.
The model-plus-length probe estimates whether the model probability adds signal
after candidate length is included; it is a diagnostic association rather than
a causal correction.

Analysis outputs are written to `<results-dir>/analysis/`:

```text
analysis.json
analysis_config.json
calibration_parameters.json
split_manifest.csv
metrics_all.csv
table_main.csv
table_crosslingual.csv
report.md
```

`table_main.csv` includes conditional source-pair bootstrap intervals for
accuracy, balanced accuracy, recall, specificity, ROC AUC, Brier score,
negative log likelihood, calibration error, and paired ranking. Cross-language
deltas use the same resampled source IDs in both languages. Exact McNemar tests
are included as descriptive instance-level checks, while the source-pair
bootstrap is the primary uncertainty estimate.

The calibrator-fit uncertainty is not included in these intervals. For a final
paper, freeze the split and hypotheses before evaluating additional models,
then confirm important findings across model families and languages. Analysis
performed after inspecting a run must be described as exploratory.

To reuse a frozen split, pass the manifest produced by the first analysis:

```bash
python scripts/analyze_detection_results.py \
  --results-dir results/detection/<experiment_name> \
  --split-manifest results/detection/<frozen_run>/analysis/split_manifest.csv
```

The analyzer rejects manifests with missing, extra, duplicate, or invalid
sample assignments. Cross-language McNemar p-values also include Holm
adjustments across the reported analysis methods.

## Detection Outputs

Results are written under:

```text
results/detection/<experiment_name>/
  config.json
  summary_all.json
  datasets/
    en.jsonl
    en.audit.csv
    en.meta.json
    en.quality.json
    zh.jsonl
    zh.audit.csv
    zh.meta.json
    zh.quality.json
  <model>/<language>/
    summary.json
    predictions.jsonl
    paired_audit.csv
```

The dataset cache stores source and translated fields together, a stable source
ID, translation token counts, and field-level quality warnings. A cache is
reused only when its translator settings and source-data fingerprint match.
Changing the translator, source data, translation policy, or review manifest
therefore rebuilds the Chinese cache automatically.

`datasets/<language>.quality.json` reports missing numbers, repeated output,
lost question intent, low-coverage or empty sentence translations, truncation,
empty output, and loss of grounded-answer alignment. It separates initial
automatic warnings from unresolved warnings after recorded review decisions.
Preserved English entity names and numeric equivalents count as valid aliases.
Numeric checks normalize thousands separators, common Chinese numerals, decade
expressions, and Chinese large-number units. Concatenated translated ranges such
as `19921998` are deterministically repaired to `1992-1998`. Punctuation-only
outputs and newly repeated clauses are critical failures.
`datasets/<language>.audit.csv` includes original machine translations, final
reviewed translations, strategies, review reasons, and knowledge segments.
Smoke configs record gate failures without aborting; the full config makes the
gate fatal.

Detection summaries report:

- generated, excluded, and primary-metric sample counts
- accuracy and balanced accuracy
- hallucination precision, recall, and F1
- specificity and Matthews correlation coefficient
- confusion matrix
- Brier score, negative log likelihood, and expected calibration error
- ROC AUC and paired-candidate ranking accuracy
- strict output-format compliance
- high-confidence errors
- predicted-hallucination rate
- candidate-length-only ROC AUC and probability correlation
- performance on a one-to-one length-matched subset

Official classification metrics use the model's generated `YES`/`NO` response.
First-token probabilities are retained for confidence, calibration, ROC AUC,
and paired ranking. `paired_audit.csv` places both candidates and predictions
on one row for manual review, including any source-review exclusion and reason.

`scripts/run_crosslingual.py` is retained as an exploratory open-ended
generation runner. Its substring-based `correct`, `hallucinated`, and `neither`
rates must not be used as final paper results without a validated semantic
judge.

Both experiment runners accept a structured NLLB translator configuration:

```json
{
  "translator": {
    "model": "facebook/nllb-200-distilled-600M",
    "source_lang": "eng_Latn",
    "target_lang": "zho_Hans",
    "dtype": "float16",
    "num_beams": 4
  }
}
```

## Research Direction

Recommended focused paper angle:

> Cross-lingual hallucination and refusal calibration in open-weight LLMs.

Core metrics:

- balanced accuracy and hallucination F1
- Matthews correlation coefficient
- decision-token confidence and calibration error
- high-confidence detection errors
- cross-lingual degradation relative to English

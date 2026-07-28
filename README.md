# Hallucinations in LLMs

Structured research code for benchmarking hallucination, token-level confidence,
and cross-lingual refusal calibration in open-weight LLMs.

## Project Layout

```text
configs/
  detection.example.json       Correct paired HaluEval detection smoke test
  experiment.example.json      Example experiment configuration
  models.example.json          Open-weight model list
  languages.example.json       Language list and prompt templates
jobs/
  submit_detection_tc2.sh      HaluEval detection job for CCDS MGPU-TC2
  submit_experiment_ntu.sh     Slurm entrypoint for NTU GPU cluster
scripts/
  prefetch_hf_assets.py        Downloads configured models and datasets
  run_halueval_detection.py    Paired multilingual hallucination detector
  run_crosslingual.py          Main cross-lingual experiment runner
  run_hallulens_task.py        HalluLens-style task runner
  aggregate_structured.py      Aggregates structured JSONL outputs
src/hallu_llm/
  data.py                      Dataset loading and sample normalization
  detection.py                 Detection instances, scoring, and metrics
  evaluation.py                Core scoring loops
  metrics.py                   Metrics and confidence helpers
  models.py                    HF model loading and generation
  prompts.py                   Prompt templates
  translation.py               Translation helpers
  translation_quality.py       Translation diagnostics and quality gate
  utils.py                     IO, seeds, filesystem helpers
```

The original single-file scripts are kept in the repository for continuity.
Use the `scripts/` entrypoints for new experiments.

## Quick Start

Install dependencies in your cluster environment:

```bash
python -m pip install torch transformers accelerate datasets sentencepiece protobuf safetensors matplotlib pandas
```

Run the 20-source-sample HaluEval detection smoke test:

```bash
python scripts/run_halueval_detection.py --config configs/detection.example.json
```

The example uses `facebook/nllb-200-distilled-600M` for English-to-Simplified
Chinese translation with the NLLB language codes `eng_Latn` and `zho_Hans`.
Four-beam deterministic decoding is used to reduce the translation corruption
seen with the earlier OPUS baseline.

The runner evaluates both the grounded and hallucinated candidate for every
source sample. The smoke test therefore produces 40 balanced decisions per
language. It asks for a strict `YES` (hallucinated) or `NO` (grounded) response
and uses that generated decision for classification. Normalized first-token
probability over those labels is retained as the confidence score.

The paired runner requires HaluEval subset `qa`. Do not change it to
`qa_samples`: that subset contains one candidate plus a `yes`/`no` label rather
than separate `right_answer` and `hallucinated_answer` fields. The loader rejects
that incompatible schema before model inference.

Prefetch the dataset, detector model, and NLLB model from a login node before
submitting an offline compute job:

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
  python scripts/prefetch_hf_assets.py --config configs/detection.example.json
```

Run the smoke test on the NTU CCDS TC2 partition:

```bash
sbatch --export=ALL,CONFIG=configs/detection.example.json jobs/submit_detection_tc2.sh
```

After manually checking the smoke-test predictions, run 150 source samples
without editing the tracked configuration:

```bash
sbatch --export=ALL,CONFIG=configs/detection.example.json,NUM_SAMPLES=150,EXPERIMENT_NAME=crosslingual_halueval_detection_nllb_150_v1 jobs/submit_detection_tc2.sh
```

Run the focused unit tests:

```bash
python -m unittest discover -s tests -v
```

## Detection Outputs

Results are written under:

```text
results/detection/<experiment_name>/
  config.json
  summary_all.json
  datasets/
    en.jsonl
    en.meta.json
    en.quality.json
    zh.jsonl
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
Changing from OPUS to NLLB therefore rebuilds the Chinese cache automatically.

`datasets/<language>.quality.json` reports missing numbers, repeated output,
lost question intent, truncation, empty output, and loss of grounded-answer
alignment between the translated answer and knowledge. The smoke config records
quality-gate failures without aborting so the audit files are available. After
validating suitable thresholds, set `fail_on_quality_gate` to `true` for full
runs so model inference cannot continue on a damaged translated dataset.

Detection summaries report:

- accuracy and balanced accuracy
- hallucination precision, recall, and F1
- specificity and Matthews correlation coefficient
- confusion matrix
- Brier score, negative log likelihood, and expected calibration error
- ROC AUC and paired-candidate ranking accuracy
- strict output-format compliance
- high-confidence errors

Official classification metrics use the model's generated `YES`/`NO` response.
First-token probabilities are retained for confidence, calibration, ROC AUC,
and paired ranking. `paired_audit.csv` places both candidates and predictions
on one row for manual review.

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

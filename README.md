# Hallucinations in LLMs

Structured research code for benchmarking hallucination, token-level confidence,
and cross-lingual refusal calibration in open-weight LLMs.

## Project Layout

```text
configs/
  detection.example.json       Selected NLLB-600M reviewed smoke test
  detection.nllb-600m.json     Explicit NLLB-600M configuration
  detection.nllb-600m.full.json
                               Strict 150-sample full-study configuration
  detection.nllb-1.3b.json     Optional matched NLLB-1.3B ablation
  halueval_reviews.json        Versioned translation and label review decisions
  experiment.example.json      Example experiment configuration
  models.example.json          Open-weight model list
  languages.example.json       Language list and prompt templates
jobs/
  submit_detection_tc2.sh      HaluEval detection job for CCDS MGPU-TC2
  submit_nllb_ablation_tc2.sh  Two-task NLLB model-size ablation
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
  text.py                      Conservative sentence-boundary repair
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

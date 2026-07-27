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

The runner evaluates both the grounded and hallucinated candidate for every
source sample. The smoke test therefore produces 40 balanced decisions per
language. It asks for a strict `YES` (hallucinated) or `NO` (grounded) response
and classifies using normalized first-token probability over those labels.

Run the smoke test on the NTU CCDS TC2 partition:

```bash
sbatch --export=ALL,CONFIG=configs/detection.example.json jobs/submit_detection_tc2.sh
```

After manually checking the smoke-test predictions, run 150 source samples
without editing the tracked configuration:

```bash
sbatch --export=ALL,CONFIG=configs/detection.example.json,NUM_SAMPLES=150,EXPERIMENT_NAME=crosslingual_halueval_detection_150 jobs/submit_detection_tc2.sh
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
    zh.jsonl
    zh.meta.json
  <model>/<language>/
    summary.json
    predictions.jsonl
```

The dataset cache stores source and translated fields together, a stable source
ID, translation token counts, and truncation warnings. A cache is reused only
when its translator settings and source-data fingerprint match.

Detection summaries report:

- accuracy and balanced accuracy
- hallucination precision, recall, and F1
- specificity and Matthews correlation coefficient
- confusion matrix
- Brier score, negative log likelihood, and expected calibration error
- strict output-format compliance
- high-confidence errors

`scripts/run_crosslingual.py` is retained as an exploratory open-ended
generation runner. Its substring-based `correct`, `hallucinated`, and `neither`
rates must not be used as final paper results without a validated semantic
judge.

For NLLB translation, the detection runner also accepts a structured translator
configuration:

```json
{
  "translator": {
    "model": "facebook/nllb-200-distilled-600M",
    "source_lang": "eng_Latn",
    "target_lang": "zho_Hans"
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

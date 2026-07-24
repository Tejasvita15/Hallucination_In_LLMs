# Hallucinations in LLMs

Structured research code for benchmarking hallucination, token-level confidence,
and cross-lingual refusal calibration in open-weight LLMs.

## Project Layout

```text
configs/
  experiment.example.json      Example experiment configuration
  models.example.json          Open-weight model list
  languages.example.json       Language list and prompt templates
jobs/
  submit_experiment_ntu.sh     Slurm entrypoint for NTU GPU cluster
scripts/
  run_crosslingual.py          Main cross-lingual experiment runner
  run_hallulens_task.py        HalluLens-style task runner
  aggregate_structured.py      Aggregates structured JSONL outputs
src/hallu_llm/
  data.py                      Dataset loading and sample normalization
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

Run a cross-lingual HaluEval-style experiment:

```bash
python scripts/run_crosslingual.py --config configs/experiment.example.json
```

Run on Slurm:

```bash
sbatch --export=ALL,CONFIG=configs/experiment.example.json jobs/submit_experiment_ntu.sh
```

## Research Direction

Recommended focused paper angle:

> Cross-lingual hallucination and refusal calibration in open-weight LLMs.

Core metrics:

- accuracy
- hallucination rate
- refusal / neither rate
- mean token confidence for correct, hallucinated, and neither outputs
- cross-lingual degradation relative to English


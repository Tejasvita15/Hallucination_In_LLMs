# Hallucination Evaluation and Inference-Time Mitigation

This repository accompanies a master's dissertation investigating factual reliability in three open-weight language models. It brings together four benchmark datasets, token-level generation traces, researcher-labelled answers, a paired English–Chinese study and inference-time mitigation experiments.

The research examines **when models make reliable factual decisions, whether token likelihood distinguishes supported answers, and how prompting and decoding interventions affect performance**. Comparisons use matched inputs within each experiment and task-specific metrics, with explicit reporting of output validity.

## Principal findings and contributions

- **Comparative benchmark evaluation:** Llama achieves the highest observed English HaluEval review accuracy and multilingual Mu-SHROOM span IoU. Qwen leads the official TruthfulQA evaluation and RAGTruth balanced accuracy and positive-only span overlap. These results identify distinct strengths across tasks.
- **Measured benefit from claim checking:** Llama's TruthfulQA generated-choice accuracy increases from **58.14% to 65.73%**, a gain of **7.59 percentage points** with an exploratory paired 95% bootstrap interval of **4.90–10.16 points**.
- **Evidence about token-confidence behaviour:** Across the 90-answer researcher-labelled pilot, unsupported answers have higher mean selected-token probability within each model. This demonstrates that high token likelihood can accompany unsupported claims and motivates evaluating confidence against semantic judgments.
- **Paired cross-lingual analysis:** Qwen's English candidate-review accuracy exceeds Chinese by **6.38 percentage points** on the primary matched sample. Class-specific results reveal increased hallucination recall but reduced grounded recall in Chinese, alongside translation and validator effects.
- **Evaluation methodology:** The analysis separates correctness, output validity, span localization and token likelihood. Strict and recovery results expose the impact of output conventions, while overall and positive-only overlap reveal sensitivity to metric definitions.
- **Traceable experimental evidence:** The project analyses **30,000 English generation traces** and an English mitigation suite of **99,876 outputs across 39 shards**, supported by saved predictions, audit summaries and reproducible plotting code.

The contribution is an empirical characterization of model and intervention behaviour under documented conditions. Detailed results and the scope of each finding are provided below.

## Models

| Name | Exact checkpoint |
|---|---|
| Llama | `meta-llama/Llama-3.1-8B-Instruct` |
| Qwen | `Qwen/Qwen2.5-7B-Instruct` |
| DeepSeek | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` |

The result alias `deepseek-r1-7b` denotes the distilled checkpoint, not full DeepSeek-R1.

## Repository layout

- `src/hallu_llm/`: model loading, prompts, evaluation, decoding and metrics.
- `scripts/`: preparation, execution, auditing and figure generation.
- `configs/`: versioned experiment configurations.
- `jobs/`: existing scheduler adapters. Site-specific paths are historical compatibility names, not portable resource specifications.
- `tests/`: implementation checks.
- `analysis/inputs/`: stable filenames for audited benchmark, mitigation, paired-comparison and clarified human-review summaries.
- `analysis/assets/`: generated figures, tables, captions, provenance manifest and browser gallery.
- `report/`: local dissertation sources and historical evidence records. Historical paths remain unchanged to preserve traceability.

## Installation and analysis

Use Python 3.10 or later. Model inference requires a CUDA-compatible PyTorch installation with sufficient GPU memory. Install PyTorch appropriate to the execution environment before installing the remaining dependencies.

```bash
python -m pip install -r requirements.txt
python scripts/build_thesis_assets.py
```

For figure generation alone, a lightweight environment is sufficient:

```bash
python -m pip install matplotlib==3.10.7 numpy
python scripts/build_thesis_assets.py
```

Input hashes and the script hash are recorded in `manifest.json`.

`requirements-truthfulqa.txt` is intentionally separate: the official likelihood evaluation uses a compatible pinned harness/Transformers stack in an isolated environment. `requirements-thesis.txt` supports the existing benchmark preparation adapter. These files are not redundant substitutes for the main requirements.
## Experimental scope

| Experiment | Scope |
|---|---|
| English HaluEval review | 20,000 candidates from 10,000 sources per model |
| Official TruthfulQA | 817 questions per model, likelihood-based MC1/MC2 |
| RAGTruth detection | 2,675 existing responses from 450 sources per model |
| Mu-SHROOM detection | 1,902 responses across 14 languages per model |
| English confidence traces | 10,000 generated answers per model |
| Human confidence pilot | 30 matched sources, 90 answers, one researcher |
| Qwen bilingual candidate review | 300 candidates per language; 282 after paired exclusions |
| Qwen bilingual generation | 150 matched sources per language |
| English mitigation | 99,876 outputs over 39 shards |


## Results and interpretation

### English candidate review

| Model | Accuracy % | Coverage % | Grounded recall % | Hallucination recall % |
|---|---:|---:|---:|---:|
| Llama | 79.935 | 99.925 | 78.32 | 81.55 |
| Qwen | 79.395 | 99.690 | 71.62 | 87.17 |
| DeepSeek | 76.890 | 97.015 | 76.45 | 77.33 |

Invalid decisions count as incorrect. Llama minus Qwen is +0.540 percentage points, with an exploratory source-cluster interval of [-0.055, 1.150]. This does not establish a clear advantage. Exact-quote flags number 1,991, 4,378 and 6,446 respectively; these indicate protocol compliance, not necessarily incorrect verdicts.

### Official TruthfulQA

| Model | MC1 % | MC2 % |
|---|---:|---:|
| Qwen | 47.74 | 64.82 |
| Llama | 37.21 | 53.99 |
| DeepSeek | 28.52 | 45.63 |

Qwen minus Llama is +10.53 points on MC1 (95% paired interval 7.47–13.59) and +10.82 on MC2 (8.27–13.26). These are completion-prompt likelihood results, not free-form answer accuracy or prompted-choice mitigation scores.

### Span detection

| RAGTruth model | Accuracy % | Balanced accuracy % | Coverage % | Character F1 | Positive-only IoU |
|---|---:|---:|---:|---:|---:|
| Qwen | 68.11 | 59.92 | 100.00 | 0.1762 | 0.2110 |
| Llama | 41.20 | 54.13 | 99.48 | 0.1271 | 0.1496 |
| DeepSeek | 44.71 | 40.88 | 66.32 | 0.0835 | 0.0508 |

Always-negative accuracy is 64.75%. Qwen recalls 32.13% of hallucinated responses; Llama recalls 97.99% but has only 10.28% specificity. Overall IoU ranks DeepSeek highest because valid empty/empty matches receive one; positive-only IoU reverses this conclusion. Neither measure should be reported without its definition.

Mu-SHROOM mean hard-span IoU is 0.3009 for Llama, 0.2482 for Qwen and 0.1841 for DeepSeek; coverage is 86.54%, 76.81% and 69.19%. Its 1,845 positive and 57 negative responses make raw response accuracy misleading. Evaluation is closed-book; URLs are metadata rather than retrieved evidence.

### Token confidence and human assessment

The 30,000 English traces pass the recorded arithmetic and alignment checks. Complete-answer mean token probabilities are 0.8822 (Llama), 0.9611 (Qwen) and 0.9356 (DeepSeek). These are not calibrated factuality probabilities.

| Model | Supported | Unsupported | Uncertain | Supported mean probability | Unsupported mean probability | Error AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Llama | 21 | 4 | 5 | 0.8602 | 0.8858 | 0.4643 |
| Qwen | 23 | 3 | 4 | 0.9685 | 0.9848 | 0.3333 |
| DeepSeek | 21 | 5 | 4 | 0.9373 | 0.9471 | 0.4286 |

The clarified single-researcher sample contains 65 supported, 12 unsupported and 13 uncertain answers. Binary comparisons exclude uncertainty and use negative mean probability as the error score. Unsupported means are higher within each model, but small groups do not support significance, calibrated-overconfidence or population-ranking claims. No inter-annotator agreement is asserted.

### Paired English–Chinese study

Qwen candidate-review accuracy is 80.50% in English and 74.11% in Chinese on 282 scored candidates per language. The +6.38-point English advantage has an exploratory source-cluster interval of 1.06–11.70. Chinese increases hallucination recall (90.07% to 97.16%) but decreases grounded recall (70.92% to 51.06%). All-candidate accuracy is 78.33% versus 73.00%.

Translation interventions and nine excluded sources limit interpretation. A whitespace-based explanation rule flags Chinese disproportionately without invalidating recognized verdicts. Separate bilingual generation means (probability 0.9622/0.9089; tokens 33.03/55.23) describe output distributions, not a factuality gap. Independent bilingual semantic equivalence is not established.

### Mitigation

| TruthfulQA model | Baseline strict % | Claim-check strict % | Baseline recovery % | Claim-check recovery % |
|---|---:|---:|---:|---:|
| Llama | 58.14 | 65.73 | 58.14 | 65.73 |
| Qwen | 65.61 | 68.05 | 65.61 | 68.18 |
| DeepSeek | 7.34 | 0.00 | 43.21 | 44.68 |

Llama's paired improvement is +7.59 points (95% interval 4.90–10.16). Qwen's strict improvement is +2.45 (-0.12–5.14). DeepSeek's strict failures are strongly parser-dependent; recovered improvement is +1.47 (-2.08–5.02). Recovery is post-hoc, rejects ambiguous/truncated outputs and never consults the gold label to select a decision.

Mu-SHROOM English strict character F1 changes from 0.0137 to 0.0787 for Llama, 0.0031 to 0.0111 for Qwen and 0.0050 to 0 for DeepSeek; absolute localization remains weak. These are detection outcomes.

Source-grounded generation compares baseline, evidence-first and CAD with fixed alpha=1 and a 2,048-token budget. CAD retains high structural validity; DeepSeek evidence-first invalidity is 91.97% on HaluEval and 94.89% on RAGTruth. Format compliance and token counts do not establish semantic hallucination reduction. Original candidate labels cannot label newly generated answers.

## Running and verifying experiments

Use the entrypoints in `scripts/`; their `--help` lists task-specific arguments.

```bash
python scripts/run_truthfulqa_mc.py --help
python scripts/run_external_benchmark.py --help
python scripts/run_halueval_detection.py --help
python scripts/submit_mitigation_suite.py
```

The final command displays the mitigation submission plan by default. Review scheduler configuration before explicitly submitting jobs. No inference or new jobs are triggered by building figures.

To audit a retained mitigation archive:

```bash
python scripts/analyze_mitigation_suite_archive.py --archive /path/to/mitigation_suite_en_v2.zip --output analysis/mitigation_audit
```

Reported runs used a CUDA-enabled GPU environment. Original run metadata retains the actual hardware and software configuration; provider and access arrangements are not needed to interpret this repository. CPU plotting is independent of the inference hardware.

## Verification and reproducibility

Reported scores were checked against the retained predictions, and token summaries were recomputed from saved traces. The mitigation audit examined all **99,876 prediction rows across 39 shards**, with no discrepancies in the integrity conditions tested. These checks support agreement between the recorded outputs and the reported calculations. The analysis scripts and plotting manifest provide a traceable route from retained evidence to tables and figures.

The verification covers saved-output analysis; it does not constitute a second, independent execution of every model forward pass. Original configurations, hashes, exclusion records and annotation versions are retained so that analysis choices can be inspected.

## Scope of interpretation

The findings apply to the evaluated checkpoints, inputs and protocols. Their interpretation follows four principles:

1. **Match the conclusion to the task.** Likelihood-based answer ranking, generated choices, candidate review and span detection measure different capabilities. Claim-checking improvements are attributed to the task on which they were observed. CAD generation results characterize structural validity and output behaviour; semantic reduction in unsupported generation requires answer-specific labels.
2. **Report uncertainty with the comparison.** Paired bootstrap intervals retain the specified question or source grouping. They are exploratory and are not adjusted for multiple comparisons; repeated outputs from one source are not treated as additional independent source questions.
3. **Keep semantic evidence explicit.** The English confidence pilot uses one researcher and small unsupported-answer groups. The bilingual comparison includes translation interventions and has no independent bilingual adjudication. These design boundaries define how broadly the findings can be generalized.
4. **Distinguish analysis artifacts from restart checkpoints.** The prediction files support the reported analysis. Some checkpoint copies in the supplied mitigation archive are empty, so that archive alone is insufficient for complete job resumption. This is a backup-completeness issue, distinct from the verified calculations on retained predictions.

Together, the results support task-specific conclusions about factual decisions, confidence behaviour, language sensitivity and mitigation. They provide a basis for further semantic validation and controlled extensions, with the evidence for each claim made explicit.

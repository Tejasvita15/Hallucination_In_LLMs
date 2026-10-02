# Full English mitigation suite audit

Checked 99,876 prediction rows across 39 shards. Integrity discrepancies: 0.

Coverage, original source content, prompts, source/generation hashes, saved code hashes (LF-normalized), available nonempty checkpoint manifests, token counts/means and directly scoreable labels were checked. Full token-vocabulary distributions and per-row duplicate checkpoint files were not independently replayed. Original archive and metrics were not changed.

Archive limitation: 91,770 checkpoint records and 33 checkpoint manifests are zero bytes. None of the 39 prediction files, summaries or run-metadata files are empty. The predictions remain analyzable, but this ZIP is not a complete resumable checkpoint backup. Preserve the original cluster directory and verify it before any cleanup. The cause of empty checkpoint files is unknown.

## TruthfulQA: prompted-choice accuracy

These are generated-choice results, not official likelihood-based MC1/MC2. Invalid outputs count as incorrect. Claim checking requires a CHECK preface and FINAL heading in the strict protocol. Missing structure can dominate results, especially for DeepSeek.

| Model | Baseline strict | Claim-check strict | Baseline recovery | Claim-check recovery |
|---|---:|---:|---:|---:|
| deepseek-r1-7b | 7.34% | 0.00% | 43.21% | 44.68% |
| llama3.1-8b | 58.14% | 65.73% | 58.14% | 65.73% |
| qwen2.5-7b | 65.61% | 68.05% | 65.61% | 68.18% |

Recovery is a separately reported post-hoc sensitivity analysis. It rejects truncated outputs and ambiguous declarations and does not consult correct labels. It cannot be presented as prespecified or substituted silently for strict metrics.

| Model / analysis | Claim-check minus baseline | Paired 95% interval |
|---|---:|---:|
| deepseek-r1-7b / strict | -7.34 pp | [-9.18, -5.51] pp |
| deepseek-r1-7b / posthoc_recovery | +1.47 pp | [-2.08, +5.02] pp |
| llama3.1-8b / strict | +7.59 pp | [+4.90, +10.16] pp |
| llama3.1-8b / posthoc_recovery | +7.59 pp | [+4.90, +10.16] pp |
| qwen2.5-7b / strict | +2.45 pp | [-0.12, +5.14] pp |
| qwen2.5-7b / posthoc_recovery | +2.57 pp | [+0.00, +5.26] pp |

Intervals use 2,000 paired question resamples and are exploratory, without multiple-comparison correction.

## Mu-SHROOM English: span detection

These measure detector quality, not fewer hallucinations in newly generated answers. Strict JSON and offset failures count as invalid. Scores are low in absolute terms.

| Model / method | Strict character F1 | Recovery character F1 | Strict valid / 154 |
|---|---:|---:|---:|
| deepseek-r1-7b / baseline | 0.0050 | 0.0150 | 77 |
| deepseek-r1-7b / claim_check | 0.0000 | 0.0235 | 0 |
| llama3.1-8b / baseline | 0.0137 | 0.0137 | 86 |
| llama3.1-8b / claim_check | 0.0787 | 0.0787 | 132 |
| qwen2.5-7b / baseline | 0.0031 | 0.0031 | 127 |
| qwen2.5-7b / claim_check | 0.0111 | 0.0172 | 113 |

## Source-grounded generation: factuality pending

HaluEval and RAGTruth outputs still require semantic annotation. Existing dataset labels refer to old candidate responses and cannot label these new generations. No generation-level hallucination reduction is established by this audit.

| Dataset / model / method | Generated | Format/reasoning invalid | Truncated | Mean output tokens |
|---|---:|---:|---:|---:|
| halueval_qa / deepseek-r1-7b / baseline | 10000 | 71 | 71 | 322.4 |
| halueval_qa / deepseek-r1-7b / cad | 10000 | 74 | 67 | 269.4 |
| halueval_qa / deepseek-r1-7b / evidence_first | 10000 | 9197 | 48 | 302.8 |
| halueval_qa / llama3.1-8b / baseline | 10000 | 0 | 0 | 17.2 |
| halueval_qa / llama3.1-8b / cad | 10000 | 1 | 1 | 19.2 |
| halueval_qa / llama3.1-8b / evidence_first | 10000 | 45 | 5 | 96.2 |
| halueval_qa / qwen2.5-7b / baseline | 10000 | 0 | 0 | 13.8 |
| halueval_qa / qwen2.5-7b / cad | 10000 | 0 | 0 | 15.8 |
| halueval_qa / qwen2.5-7b / evidence_first | 10000 | 2 | 0 | 88.0 |
| ragtruth / deepseek-r1-7b / baseline | 450 | 0 | 0 | 643.2 |
| ragtruth / deepseek-r1-7b / cad | 450 | 1 | 1 | 566.1 |
| ragtruth / deepseek-r1-7b / evidence_first | 450 | 427 | 0 | 632.4 |
| ragtruth / llama3.1-8b / baseline | 450 | 0 | 0 | 149.8 |
| ragtruth / llama3.1-8b / cad | 450 | 0 | 0 | 162.2 |
| ragtruth / llama3.1-8b / evidence_first | 450 | 0 | 0 | 454.2 |
| ragtruth / qwen2.5-7b / baseline | 450 | 0 | 0 | 136.5 |
| ragtruth / qwen2.5-7b / cad | 450 | 0 | 0 | 126.7 |
| ragtruth / qwen2.5-7b / evidence_first | 450 | 9 | 0 | 254.6 |

## Next steps

Preserve strict results and report format adherence separately from factual correctness. Review post-hoc recovery examples before using sensitivity results. Predefine and annotate a manageable source-matched, model/condition-masked sample of HaluEval/RAGTruth outputs; do not choose cases by favorable confidence or correctness. Do not rerun all experiments merely to repair formatting. If a new prompt/format protocol is tested, version it as a new experiment and retain this run. Hindi remains deferred.

# Run English mitigation first

This is a new, exploratory Qwen answer-generation experiment. It preserves every earlier result and does not depend on Hindi translation review.

From the repository root on ASPIRE2A+, after pulling this commit:

```bash
qsub jobs/nscc_2ap/english_mitigation_v1.pbs
```

The PBS file uses project 13014407, queue aiq1 (as in the supplied successful allocation log), one GPU and a 24-hour limit. It uses the existing settings.local.sh, conda environment and cached Qwen model. No dependency reinstall, smoke job or online download is performed. Queue acceptance and actual H100 execution must be confirmed on ASPIRE; they have not been tested locally.

## What runs

150 matched English questions × baseline / evidence-first / context-aware decoding = 450 answers. The 30 separate development sources are retained but not run or used to select parameters in v1. CAD alpha is fixed at 1.0 before outputs. This English-first decision supersedes the earlier proposed alpha search in the Hindi extension plan.

All three conditions generate a fresh matched baseline using the same Qwen revision, tokenizer, 512-token output budget and manual greedy decoding implementation. Unlike earlier model.generate runs, the decoder applies no inherited generation penalties, temperature scaling or candidate filter; old baseline outputs are not substituted. It stops on the model's EOS IDs. Input truncation is prohibited; output truncation is recorded as invalid. Evidence-first outputs must have one EVIDENCE: and one ANSWER: heading in the correct order. Evidence text is separate from the final answer and requires its own qualitative audit.

CAD uses two independent KV caches and the same generated prefix, with supplied knowledge present in one prompt and an empty knowledge field in the other. The next-token score is (1+alpha) log p_with − alpha log p_without. This is a task-specific implementation of the method described in [Trusting Your Evidence](https://arxiv.org/abs/2305.14739), not an exact reproduction of that paper's experiments. The answer and abstention instructions are otherwise identical to baseline. References are retained for later annotation but never enter either prompt.

Raw contextual model probabilities and normalized CAD probabilities are stored separately. Neither is a probability of factual correctness. Generation time includes both CAD branches; forward-pass counts and evidence tokens are recorded. Method order rotates across sources to reduce systematic timing-order effects. Runtime comparisons still have hardware/load uncertainty.

## Outputs to return

Folder: results/english_mitigation_v1/qwen2.5-7b/en/

- predictions.jsonl: all 450 answers, final-answer parsing, raw/decoding token traces and provenance hashes.
- summary.json: completion and invalid/truncation counts. It deliberately contains no unverified factuality metrics.
- run_metadata.json: code hashes, package versions, model/tokenizer revisions and configuration.
- checkpoint/: per-answer atomic records for resuming the same unchanged run.

Return the first three files after completion. If interrupted, resubmit the same PBS file. A hard scheduler kill may leave checkpoint/writer.lock; remove only that file after verifying the original job is no longer running. Do not delete checkpoints or override provenance mismatches to force a resume.

## Analysis after generation

Create a shuffled, condition-masked researcher annotation packet. Label supported, unsupported, abstention and uncertain final answers; retain invalid/truncated rows in all-instance denominators. Record question-answering success separately. Primary metric: supported answers / all 150 questions per condition. Report unsupported rate, abstention, invalids, paired source-level bootstrap intervals and inference cost. Reference-answer string matching alone is not a factuality judge.

The same 150 sources were previously examined in the English/Chinese study, so call this exploratory. Existing ambiguous or defective sources are retained for transparent annotation, not silently fixed or dropped. No mitigation benefit is established until outputs are adjudicated. Hindi remains a subsequent extension; its generated translation packet is preserved locally but is not required for this job.

Local validation: CPU-only mathematical/cache/parser tests and syntax checks. No pretrained-model smoke run or H100 experiment was performed locally.

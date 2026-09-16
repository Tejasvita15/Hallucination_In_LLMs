# NSCC ASPIRE 2A+ (2ap)

## Current next step: evidence review v2 smoke tests

The v1 audit found bare YES/NO responses for all Qwen/Llama examples and label
confusion in DeepSeek explanations. Those runs are diagnostic, not a validated
reasoning comparison. Run these replacement smoke jobs from the repository root:

```bash
qsub jobs/nscc_2ap/qwen_review_v2.pbs
qsub jobs/nscc_2ap/llama_review_v2.pbs
qsub jobs/nscc_2ap/deepseek_review_v2.pbs
```

After all three jobs finish (including any quality-gate failures), run on the login node:

```bash
python scripts/audit_reasoned_smoke.py --version v2 --output results/detection/evidence_review_v2_audit.txt
```

Share that audit and the job logs before starting full runs.

Stage one requests a short evidence review with an exact source quotation and
SUPPORTED/UNSUPPORTED verdict, without the previous no-explanation instruction.
Stage two uses a fresh prompt containing the evidence and review, explicitly
maps UNSUPPORTED to YES and SUPPORTED to NO, and scores those two choices.
No synthetic closing think tag is added in v2. Raw label mass is retained;
constrained answer probabilities are not unconditional calibrated confidence.

Results are isolated in `results/detection/nscc_<model>_evidence_review_smoke_v2/`.
Predictions retain both prompts, the generated review, parsed review, final
decision, and `review_quality_flags`. Missing/short explanations, invalid JSON,
quotes absent from the evidence, invalid verdicts, verdict/decision disagreements,
and truncation fail the gate after saving predictions. These are mechanical
checks, not proof that the explanation is semantically correct. Manual review
is still necessary. Flagged rows remain in diagnostic metrics rather than being
silently dropped. A failed gate is not a GPU/environment failure.

Keep full jobs on hold. The v1 full drafts and frozen production configurations
are unchanged; v2 is exploratory and must pass this new smoke test first.

Submit from the repository root. These scripts request one GPU through the
`normal` routing queue for up to 24 hours. CPU and memory allocation follow
the site's GPU ratios. They target ASPIRE 2A+, not the TC2 Slurm cluster.

Official references:
- https://help.nscc.sg/aspire2aplus/faqs/
- https://help.nscc.sg/aspire2aplus/urgent-scheduled-maintenance-for-aspire-2a-on-22-january-2025-9am-to-6-30pm-2-2-2-2-2-3-3-2/

## 1. Configure once

```bash
cp jobs/nscc_2ap/settings.example.sh jobs/nscc_2ap/settings.local.sh
nano jobs/nscc_2ap/settings.local.sh
source jobs/nscc_2ap/settings.local.sh
```

Defaults follow your supplied PBS example: project `13014407`, module
`miniforge3/24.3.0`, and cache `/scratch/users/ntu/tejasvit/huggingface`.
The LLM environment is `/scratch/users/ntu/tejasvit/conda_envs/hallucination`.
Check that the cache destination has sufficient quota. The SimVP environment is
not selected automatically. Existing `settings.local.sh` files are not updated
by Git; edit them to match these settings. Local settings are ignored
by Git. Do not put tokens in them. Do not modify settings while jobs are queued.

The Conda environment must contain the project requirements and GPU-enabled
PyTorch compatible with the H100 nodes. The scripts do not install or replace
PyTorch. They clear Python path overrides, disable user-site packages, and test
an actual CUDA operation before running inference.

## 2. Prepare on the login node

Activate the selected environment and authenticate with `hf auth login`, using
the same HF_HOME as in local settings. Llama requires approved model access.
Then run:

```bash
bash jobs/nscc_2ap/prepare_login.sh
```

This fetches three evaluation models, NLLB, controlled-generation inputs and
four static datasets. Keep the cache on shared storage with sufficient quota.
HalluLens's large external assets and endpoint setup are separate prerequisites;
see the main README. GPU jobs use the same cache in offline mode.

## 3. Smoke tests

```bash
source jobs/nscc_2ap/settings.local.sh
qsub -P "$NSCC_PROJECT_ID" -l walltime=02:00:00 \
  -v NUM_SAMPLES=20,EXPERIMENT_NAME=nscc_halueval_smoke \
  jobs/nscc_2ap/detection.pbs

for MODEL in qwen2.5-7b llama3.1-8b deepseek-r1-7b; do
  for BENCH in ragtruth mu_shroom truthfulqa; do
    qsub -P "$NSCC_PROJECT_ID" -l walltime=02:00:00 \
      -v BENCHMARK="$BENCH",MODEL_NAME="$MODEL",MAX_SAMPLES=20 \
      jobs/nscc_2ap/external_benchmark.pbs
  done
done
```

Inspect predictions, parsing failures and DeepSeek's short-output behavior before
full runs. Smoke tests are not final results.

## 4. Full benchmarks (after smoke completion)

### DeepSeek reasoning diagnostic

If DeepSeek produces only `Okay, so I` in the four-token smoke test, run:

```bash
qsub jobs/nscc_2ap/deepseek_smoke.pbs
```

After that diagnostic passes, run the matched Qwen and Llama diagnostics:

```bash
qsub jobs/nscc_2ap/qwen_smoke.pbs
qsub jobs/nscc_2ap/llama_smoke.pbs
```

Each job uses the same 20 source pairs, seed, 2,048-token budget and forced-choice
method as the DeepSeek diagnostic. Their separate output directories are
`results/detection/nscc_qwen_reasoned_smoke_v1/` and
`results/detection/nscc_llama_reasoned_smoke_v1/`. An already-completed DeepSeek
diagnostic does not need to be repeated. Compare saved reasoning and quality
flags as well as scores; this matched diagnostic does not change production configs.

This runs only DeepSeek on the same 20 English source pairs, with a 2,048-token
reasoning budget followed by explicitly constrained YES/NO scoring. Confidence
comes from the final decision position, not the first reasoning token. Reasoning
text, length, truncation, and raw label probability mass are saved. Forced-choice
parseability is guaranteed by construction and is not free-generation compliance.
Any truncated reasoning fails the quality gate after saving that language's
predictions. Inspect the audit before increasing the budget or running full data.

Results go to `results/detection/nscc_deepseek_reasoned_smoke_v1/`. This is an
exploratory protocol change; the frozen production configurations still use
direct generation. Do not interpret the original DeepSeek fallback metrics as
valid generated-answer performance or pool the two protocols. A fair comparison
of the new mode requires applying it consistently to the comparison models.

```bash
qsub -P "$NSCC_PROJECT_ID" jobs/nscc_2ap/detection.pbs

for MODEL in qwen2.5-7b llama3.1-8b deepseek-r1-7b; do
  for BENCH in ragtruth mu_shroom truthfulqa; do
    qsub -P "$NSCC_PROJECT_ID" \
      -v BENCHMARK="$BENCH",MODEL_NAME="$MODEL" \
      jobs/nscc_2ap/external_benchmark.pbs
  done
done
```

HaluEval evaluates all three models sequentially; external jobs run one model
each. The 24-hour request is a scheduling limit, not a runtime estimate.
External runs can resume compatible checkpoints. The paired detector does not
provide the same checkpoint-resume guarantee; inspect job completion before
relaunching. Avoid concurrent runs writing the same experiment output.

## 5. Controlled English/Chinese study

```bash
qsub -P "$NSCC_PROJECT_ID" jobs/nscc_2ap/controlled_generation.pbs
```

Wait for generation, complete both independent reviews and any adjudication,
then finalize on the login node in the configured environment:

```bash
python scripts/build_length_controlled_halueval.py \
  --config configs/controlled_benchmark.full.json --finalize-only
qsub -P "$NSCC_PROJECT_ID" \
  -v CONFIG=configs/detection.controlled.confirmatory.json,PREPARE_ONLY=1,FAIL_ON_TRANSLATION_QUALITY=1 \
  jobs/nscc_2ap/detection.pbs
```

After translation quality passes:

```bash
qsub -P "$NSCC_PROJECT_ID" jobs/nscc_2ap/controlled_detection.pbs
```

After that job finishes:

```bash
python scripts/analyze_detection_results.py --config configs/detection_analysis.controlled.json
```

## 6. HalluLens

Install upstream dependencies in a suitable environment and configure its model
endpoint and required search/judge credentials first. Download its data using
the upstream preparation workflow on the login node. Then, for each task:

```bash
qsub -P "$NSCC_PROJECT_ID" -v TASK=precisewikiqa jobs/nscc_2ap/hallulens.pbs
```

Other TASK values: `longwiki`, `mixedentities`, `generatedentities`. These jobs
use the configured endpoint, not an automatic three-model loop. Save outputs
separately for each model; do not change a shared endpoint while jobs run.
The suite aggregator still selects Qwen only.

## Monitor

## Audit before full reasoning runs

Run on the login node from the repository root:

```bash
python scripts/audit_reasoned_smoke.py
```

It writes `results/detection/reasoned_smoke_audit.txt` with token lengths,
bare YES/NO counts, missing/empty reasoning, truncation, matched pair IDs and
all saved evidence/reasoning responses for review. Upload this report for review.
The existing v1 prompt contains both a no-explanation instruction and an appended
request to analyze; the previous quality gate did not test this conflict.
Do not interpret the previous gate as evidence that reasoning was elicited.

Draft `qwen_full.pbs`, `llama_full.pbs`, and `deepseek_full.pbs` use separate
`detection.*-reasoned.full.json` configs, all 10,000 pairs and distinct outputs.
They preserve the v1 smoke protocol for traceability, but are not yet approved
for scientific full runs: review the reasoning first and revise/version the
prompt if required. DeepSeek's roughly 4-minute inference smoke suggests a full
run may exceed the 24-hour allocation. This paired runner has no resumable
checkpoint support; implement checkpointing or partitioning before that full run.
Do not submit these draft full jobs yet. The original production configs and
frozen confirmatory hypothesis have not been changed.

```bash
qstat -u "$USER"
qstat -f JOB_ID
ls -lt logs/nscc_2ap/
```

Application logs are under `logs/nscc_2ap/`; PBS also writes its merged output
in the submission directory. Setup failures before logging are in PBS output.
Scripts have LF line endings. Local syntax checks do not validate your account,
queue access, installed modules, GPU environment, or cluster runtime.

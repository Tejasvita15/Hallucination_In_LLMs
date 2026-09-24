# Full thesis runs on ASPIRE 2A+

Run from the repository root on the ASPIRE login node, after uploading the updated
repository. No smoke experiments are required or submitted. The local unit tests
exercise scoring and token arithmetic without downloading or running an LLM.

## Start here

1. Copy `jobs/nscc_2ap/settings.example.sh` to `jobs/nscc_2ap/settings.local.sh`
   if it does not already exist. Verify the conda path, HF cache path and project
   allocation. Current PBS defaults use project `13014407`, queue `normal`, one
   GPU and 24 hours, matching the existing ASPIRE files. The scheduler determines
   the allocated GPU; these directives do not explicitly request an H100 model.
2. Prepare dependencies and cached datasets/models online. This performs no LLM inference:

   ```bash
   bash jobs/nscc_2ap/prepare_thesis_login.sh
   ```

   The existing environment must have a working CUDA PyTorch build and access to
   the gated Llama weights. This installs the pinned evaluation harness, records
   the resolved environment, caches assets and prepares the full benchmark splits.
3. Submit the full runs:

   ```bash
   bash jobs/nscc_2ap/submit_thesis_full.sh
   ```

   This submits **13 jobs**, with no sample-limit flags. It records returned PBS
   IDs in `logs/nscc_2ap/submissions_*.tsv`. Do not invoke it again while the jobs
   are active. For staged scheduling, use `core`, `expansion`, or `crosslingual`
   as its argument instead of submitting `all`.

### Submit the PBS files directly

The shell submission helper only calls `qsub`; the scheduled jobs are `.pbs`
files. After preparation, you can submit them directly instead of using the
helper. Run the following once from the repository root to submit the same 13
jobs (do not also run the helper):

```bash
for model in llama3.1-8b qwen2.5-7b deepseek-r1-7b; do
  qsub -v MODEL_NAME="$model" jobs/nscc_2ap/truthfulqa_mc_full.pbs
  qsub -v MODEL_NAME="$model" jobs/nscc_2ap/token_confidence_full.pbs
  qsub -v MODEL_NAME="$model" jobs/nscc_2ap/ragtruth_full.pbs
  qsub -v MODEL_NAME="$model" jobs/nscc_2ap/mushroom_full.pbs
done
qsub jobs/nscc_2ap/qwen_crosslingual_full.pbs
```

These direct commands use the project allocation in the PBS headers. If your
allocation differs, add `-P YOUR_PROJECT_ID` to each `qsub` command. Asset
preparation runs on the login node because GPU jobs use the offline cache.

## What runs

| PBS file | Jobs | Work | Outputs |
|---|---:|---|---|
| `truthfulqa_mc_full.pbs` | 3 | Official harness MC1/MC2, complete validation split, original completion prompt without chat template | `results/truthfulqa_mc_v1/MODEL/` |
| `token_confidence_full.pbs` | 3 | Generate one source-grounded answer per prepared HaluEval QA question; preserve each emitted token's likelihood and entropy | `results/token_confidence_v1/MODEL/en/` |
| `ragtruth_full.pbs` | 3 | Full prepared test split, response and span detection, detector token traces | `results/benchmark_suite_v2/ragtruth/MODEL/` |
| `mushroom_full.pbs` | 3 | Full labelled prepared test split in all available languages; closed-book span detection | `results/benchmark_suite_v2/mu_shroom/MODEL/` |
| `qwen_crosslingual_full.pbs` | 1 | All 150 planned paired EN/ZH questions: quality-gated translations, 300 answer generations with token traces, 600 v3 candidate reviews | `results/token_confidence_qwen_paired150_v1/` and the configured detection directory |

The Qwen study is the complete **150-pair exploratory design**, not the entire
HaluEval corpus. Mu-SHROOM languages are native benchmark sets, not paired
translations of the same questions. Translation quality failures stop the job;
inspect and adjudicate them rather than disabling the gate.

The completed 60,000 English v3 candidate reviews remain the existing evidence.
The three legacy `*_review_v3_full.pbs` files have been fixed to select their full
configs, but the submission script does **not** rerun those completed experiments.

HalluLens is **not included** in the 13-job submission: its existing
`jobs/nscc_2ap/hallulens.pbs` invokes official scripts, but still requires the
official checkout's model endpoint and task data to be configured. An endpoint
has not been verified in this workspace. Do not describe HalluLens as completed.

## Checkpoints and recovery

### TruthfulQA import error: `AutoModelForVision2Seq`

The pinned harness imports an API absent from Transformers 5, even for text-only
evaluation. TruthfulQA now uses a separate virtual environment (default:
`${CONDA_ENV}-truthfulqa-v1`) with the compatibility versions in
`requirements-truthfulqa.txt`. It inherits the existing CUDA PyTorch installation
and installs its overrides locally; the base environment used by other jobs is
not modified by this repair. Do not rerun the general preparation script while
repairing this failure with other experiments active. Instead, run:

```bash
git pull origin main
bash jobs/nscc_2ap/prepare_truthfulqa_login.sh
```

After `TruthfulQA preparation complete`, resubmit only failed TruthfulQA jobs:

```bash
qsub -v MODEL_NAME=llama3.1-8b jobs/nscc_2ap/truthfulqa_mc_full.pbs
qsub -v MODEL_NAME=qwen2.5-7b jobs/nscc_2ap/truthfulqa_mc_full.pbs
qsub -v MODEL_NAME=deepseek-r1-7b jobs/nscc_2ap/truthfulqa_mc_full.pbs
```

Preparation imports the actual evaluator/model registry and loads both task
datasets on the login node. It performs no model inference. Other PBS files keep
using the existing conda environment. The cluster import check must pass before
this environment is marked ready; local syntax checks alone cannot verify it.

### Preparation error: `Repository id must be 'namespace/name'`

The pinned harness's official TruthfulQA tasks use the historical `truthful_qa`
alias, which newer Hugging Face clients reject. Both preparation and evaluation
now pass a task override selecting `truthfulqa/truthful_qa`. Official task prompts,
answer choices, and MC1/MC2 scoring remain unchanged. Pull the update and rerun
preparation; successfully cached datasets and weights are reused.

### Preparation error: `Feature type 'List' not found`

Harness 0.4.9.1 requires `datasets<4`; the preparation requirements explicitly
pin `datasets==3.6.0`. A cache built with datasets 5 can contain feature metadata
that this older reader cannot load. Thesis preparation and all five thesis PBS
launchers now use `$HF_HOME/datasets-thesis-3.6.0`, while model weights continue
to use `$HF_HOME/hub`. The existing `$HF_HOME/datasets` cache is left untouched.
Pull the fix and rerun `bash jobs/nscc_2ap/prepare_thesis_login.sh`; submit jobs
only after it prints `Preparation complete`. Use `bash`, not `ash`.

The preparation marker is stored inside the versioned dataset cache, so a marker
from the earlier shared-cache setup cannot bypass preparation for the new jobs.

The new generation, external-benchmark and MC runners lock their output and
check provenance before resuming. Generation/external results are saved after
every instance; TruthfulQA caches likelihood requests. Submit the same individual
PBS file again after a time-limit interruption, for example:

```bash
qsub -v MODEL_NAME=deepseek-r1-7b jobs/nscc_2ap/token_confidence_full.pbs
```

A hard scheduler kill can leave `writer.lock`. Confirm the original job has ended
before removing that one lock file. Changed code, data, resolved model snapshots
or dependencies require a new output directory; incompatible results are not
silently mixed. A complete `predictions.jsonl` and summary are exported when a
run finishes. Partial generations remain available in individual checkpoint files.

Full completion within one 24-hour job or within the thesis deadline has **not**
been measured. Resume unfinished jobs; inspect coverage, invalid/truncated outputs,
and translation quality before interpreting results. No ASPIRE jobs were launched
or GPU runtime validation performed by the local repair work.

## Confidence analysis and correctness verification

Per-token traces contain token IDs, pieces, chosen-token log probability,
probability, entropy in nats, special-token flags, and reasoning/final segments.
They use the processed greedy generation distribution, save no full vocabulary
matrix, and do not count reasoning tokens in final-answer means. Truncated or
missing final answers remain explicitly marked.

**Token likelihood is not the probability that an answer is factually correct.**
The new answer runner leaves semantic labels pending rather than deciding truth
from substring matches. Export a blinded review packet and fill semantic labels
(`supported`, `unsupported`, `abstention`, `uncertain`) with reviewer/rationale:

```bash
python scripts/analyze_token_confidence.py \
  --predictions results/token_confidence_v1/qwen2.5-7b/en/predictions.jsonl \
  --export-annotations qwen_annotations.jsonl --output qwen_review_status.json

python scripts/analyze_token_confidence.py \
  --predictions results/token_confidence_v1/qwen2.5-7b/en/predictions.jsonl \
  --annotations qwen_annotations.jsonl --output qwen_token_analysis.json
```

Annotations must match the exact generation hash. Pending, uncertain/abstaining
and truncated cases are reported separately. The initial analysis compares token
likelihood by reviewed label and computes discrimination AUROC; it does not
misreport raw token likelihood as calibrated answer confidence. Independent
adjudication, source-cluster uncertainty intervals and length-controlled analyses
remain thesis-analysis work after the runs, not automatically completed experiments.

RAGTruth/Mu-SHROOM token traces describe the **detector output**, not confidence
of the benchmark's original answer generator. Original Mu-SHROOM selected-token
logits are retained but are not normalized probabilities. First-step YES/NO
conditional likelihoods are diagnostic only, especially for reasoning models
whose first generated token is not their final verdict. Official MC1/MC2 scores
are separate from the custom binary TruthfulQA task.

## Repairs and local validation

Validation on 23 September 2026: **108 local unit tests passed**. Python syntax
compilation, Bash syntax checks of the shared scripts/full PBS files, and
`git diff --check` passed. These checks do not certify ASPIRE GPU runtime behavior.

- Invalid generated labels no longer fall back to logit argmax for primary accuracy.
- Invalid/truncated spans cannot score as correct empty spans. Unknown quoted spans
  and invalid offsets fail parsing. Invalid negative and positive responses count
  as incorrect; MCC is unavailable with abstentions rather than fabricated.
- Mu-SHROOM URLs are metadata rather than source text; native tokens/logits survive preparation.
- Full-review PBS files now reference full configurations; Unix line endings are enforced.
- The legacy `logit_confidence.py` and `truthfulqa_eval.py` forward to the corrected CLIs.
- External benchmark aggregation now handles every configured model separately.

TruthfulQA uses `lm-eval==0.4.9.1` and its official task definitions:
https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.9.1/lm_eval/tasks/truthfulqa

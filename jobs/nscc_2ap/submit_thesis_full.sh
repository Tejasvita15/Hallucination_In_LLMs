#!/bin/bash
# Run from the repository root on the ASPIRE login node. No smoke jobs.
set -euo pipefail
[[ -f jobs/nscc_2ap/settings.local.sh ]] || { echo 'Create jobs/nscc_2ap/settings.local.sh from settings.example.sh first.' >&2; exit 2; }
source jobs/nscc_2ap/settings.local.sh
: "${NSCC_PROJECT_ID:?Set NSCC_PROJECT_ID}"
: "${HF_HOME:?Set HF_HOME}"
: "${CONDA_ENV:?Set CONDA_ENV}"
GROUP=${1:-all}
case "$GROUP" in all|core|expansion|crosslingual) ;; *) echo 'Usage: bash jobs/nscc_2ap/submit_thesis_full.sh [all|core|expansion|crosslingual]' >&2; exit 2;; esac
command -v qsub >/dev/null
[[ -f "$HF_HOME/thesis_prepared.json" ]] || { echo 'Run bash jobs/nscc_2ap/prepare_thesis_login.sh first.' >&2; exit 2; }
if [[ "$GROUP" == all || "$GROUP" == core ]]; then
  [[ -s data/benchmarks/prepared/halueval_qa.jsonl ]] || { echo 'Missing prepared HaluEval data' >&2; exit 2; }
fi
if [[ "$GROUP" == all || "$GROUP" == expansion ]]; then
  for dataset in ragtruth mu_shroom; do
    [[ -s "data/benchmarks/prepared/$dataset.jsonl" ]] || { echo "Missing $dataset data" >&2; exit 2; }
  done
fi
mkdir -p logs/nscc_2ap
LOG="logs/nscc_2ap/submissions_$(date +%Y%m%d_%H%M%S).tsv"
submit() {
  local file=$1 model=${2:-} job
  if [[ -n "$model" ]]; then
    job=$(qsub -P "$NSCC_PROJECT_ID" -v "MODEL_NAME=$model" "jobs/nscc_2ap/$file.pbs")
  else
    job=$(qsub -P "$NSCC_PROJECT_ID" "jobs/nscc_2ap/$file.pbs")
  fi
  printf '%s\t%s\t%s\n' "$job" "$file" "$model" | tee -a "$LOG"
}
for model in llama3.1-8b qwen2.5-7b deepseek-r1-7b; do
  if [[ "$GROUP" == all || "$GROUP" == core ]]; then
    submit truthfulqa_mc_full "$model"
    submit token_confidence_full "$model"
  fi
  if [[ "$GROUP" == all || "$GROUP" == expansion ]]; then
    submit ragtruth_full "$model"
    submit mushroom_full "$model"
  fi
done
if [[ "$GROUP" == all || "$GROUP" == crosslingual ]]; then
  submit qwen_crosslingual_full
fi
echo "Submission record: $LOG. Do not resubmit while these jobs are active."

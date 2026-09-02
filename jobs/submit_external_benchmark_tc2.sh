#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --job-name=halluc-bench
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs results/benchmark_suite

module load anaconda
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-master_env}"

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

CONFIG="${CONFIG:-configs/benchmark_suite.full.json}"
BENCHMARK="${BENCHMARK:?Set BENCHMARK to ragtruth, mu_shroom, or truthfulqa}"
MODEL_NAME="${MODEL_NAME:-qwen2.5-7b}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
RESTART="${RESTART:-0}"

CMD=(
  python scripts/run_external_benchmark.py
  --config "$CONFIG"
  --benchmark "$BENCHMARK"
  --model-name "$MODEL_NAME"
)
if [ -n "$MAX_SAMPLES" ]; then
  CMD+=(--max-samples "$MAX_SAMPLES")
fi
if [ "$RESTART" = "1" ]; then
  CMD+=(--restart)
fi

echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Benchmark: $BENCHMARK"
echo "Command: ${CMD[*]}"
"${CMD[@]}"
echo "Finished: $(date)"

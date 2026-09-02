#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --job-name=controlled-negatives
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs results/benchmark

module load anaconda
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-master_env}"

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

CONFIG="${CONFIG:-configs/controlled_benchmark.full.json}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
RESTART="${RESTART:-0}"

CMD=(python scripts/build_length_controlled_halueval.py --config "$CONFIG")
if [ -n "$NUM_SAMPLES" ]; then
  CMD+=(--num-samples "$NUM_SAMPLES")
fi
if [ -n "$EXPERIMENT_NAME" ]; then
  CMD+=(--experiment-name "$EXPERIMENT_NAME")
fi
if [ "$RESTART" = "1" ]; then
  CMD+=(--restart)
fi

echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Command: ${CMD[*]}"
"${CMD[@]}"
echo "Finished: $(date)"

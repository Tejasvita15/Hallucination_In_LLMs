#!/bin/bash
#SBATCH --job-name=xling-hallu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs results/structured

module purge || true
module load cuda/12.1 || true

if [ -n "${CONDA_ENV:-}" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

CONFIG="${CONFIG:-configs/experiment.example.json}"

echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Config: $CONFIG"

python scripts/run_crosslingual.py --config "$CONFIG"

echo "Finished: $(date)"

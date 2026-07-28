#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --job-name=nllb-ablation
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs results/detection

module load anaconda
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-master_env}"

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

CONFIGS=(
  "configs/detection.nllb-600m.json"
  "configs/detection.nllb-1.3b.json"
)
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Config: $CONFIG"

python scripts/run_halueval_detection.py --config "$CONFIG"

echo "Finished: $(date)"

#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --job-name=controlled-detect
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

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

CONFIG="${CONFIG:-configs/detection.controlled.confirmatory.json}"

echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Config: $CONFIG"
python scripts/run_halueval_detection.py \
  --config "$CONFIG" \
  --fail-on-translation-quality
echo "Finished: $(date)"

#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --job-name=hallulens
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs results/hallulens

module load anaconda
eval "$(conda shell.bash hook)"
conda activate master_env

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

TASK="${TASK:-nonexistent}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_NAME="${MODEL_NAME:-qwen2.5-7b}"
JUDGE_MODEL="${JUDGE_MODEL:-}"
DATA="${DATA:-}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
OUTPUT_DIR="${OUTPUT_DIR:-results/hallulens}"
DTYPE="${DTYPE:-bfloat16}"

if [ "${TRANSFORMERS_OFFLINE:-0}" = "1" ] && [ ! -d "$MODEL" ]; then
  echo "TRANSFORMERS_OFFLINE=1, but MODEL is not a local directory: $MODEL" >&2
  echo "Use a downloaded model folder, for example:" >&2
  echo "  sbatch --export=ALL,MODEL=/home/msai/tejasvit001/models/Llama-3.1-8B-Instruct,MODEL_NAME=llama3.1-8b submit_hallulens_tc2.sh" >&2
  exit 2
fi

CMD=(python hallulens_eval.py
  --task "$TASK"
  --model "$MODEL"
  --model-name "$MODEL_NAME"
  --output-dir "$OUTPUT_DIR"
  --dtype "$DTYPE"
  --device-map auto
)

if [ -n "$DATA" ]; then
  CMD+=(--data "$DATA")
elif [ "$TASK" = "nonexistent" ]; then
  CMD+=(--make-nonexistent-samples --num-samples "$NUM_SAMPLES")
else
  echo "DATA must point to a JSONL file for TASK=$TASK" >&2
  exit 2
fi

if [ -n "$JUDGE_MODEL" ]; then
  CMD+=(--judge-model "$JUDGE_MODEL")
fi

if [ -n "$MAX_SAMPLES" ]; then
  CMD+=(--max-samples "$MAX_SAMPLES")
fi

echo "Running on $(hostname)"
echo "Started at $(date)"
echo "Command: ${CMD[*]}"
"${CMD[@]}"
echo "Finished at $(date)"

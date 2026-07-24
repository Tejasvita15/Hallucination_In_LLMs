#!/bin/bash
#SBATCH --job-name=hallulens
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/hallulens_%j.out
#SBATCH --error=logs/hallulens_%j.err

set -euo pipefail

mkdir -p logs results/hallulens

# Edit these for your NTU environment if the module names differ.
module purge || true
module load cuda/12.1 || true

# If you use conda on the cluster, set CONDA_ENV before sbatch:
#   sbatch --export=ALL,CONDA_ENV=myenv submit_hallulens_ntu.sh
if [ -n "${CONDA_ENV:-}" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

python -m pip install --user --upgrade \
  torch transformers accelerate datasets sentencepiece protobuf safetensors

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

# Choose a task:
#   precisewikiqa requires JSONL rows with question/gold_answer or prompt/answer.
#   longwiki requires JSONL rows with question plus reference/references/wiki_document.
#   nonexistent can use --make-nonexistent-samples for a quick paper-style refusal test.
TASK="${TASK:-nonexistent}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_NAME="${MODEL_NAME:-qwen2.5-7b}"
JUDGE_MODEL="${JUDGE_MODEL:-}"
DATA="${DATA:-}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
OUTPUT_DIR="${OUTPUT_DIR:-results/hallulens}"

CMD=(python hallulens_eval.py
  --task "$TASK"
  --model "$MODEL"
  --model-name "$MODEL_NAME"
  --output-dir "$OUTPUT_DIR"
  --dtype bfloat16
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

echo "Running: ${CMD[*]}"
"${CMD[@]}"

#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=hallulens
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"
mkdir -p logs results/benchmark_suite/hallulens

module load anaconda
eval "$(conda shell.bash hook)"
conda activate "${HALLULENS_CONDA_ENV:-master_env}"

HALLULENS_DIR="${HALLULENS_DIR:-$PROJECT_DIR/data/benchmarks/raw/HalluLens}"
TASK="${TASK:?Set TASK to precisewikiqa, longwiki, mixedentities, or generatedentities}"

case "$TASK" in
  precisewikiqa) SCRIPT="scripts/task1_precisewikiqa.sh" ;;
  longwiki) SCRIPT="scripts/task2_longwiki.sh" ;;
  mixedentities) SCRIPT="scripts/task3-1_mixedentities.sh" ;;
  generatedentities) SCRIPT="scripts/task3-2_generatedentities.sh" ;;
  *) echo "Unsupported TASK: $TASK" >&2; exit 2 ;;
esac

if [ ! -f "$HALLULENS_DIR/$SCRIPT" ]; then
  echo "Missing $HALLULENS_DIR/$SCRIPT; prepare the suite first." >&2
  exit 2
fi

echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Official HalluLens checkout: $HALLULENS_DIR"
echo "Task script: $SCRIPT"
echo "The checkout's utils/lm.py must already point to your model endpoint."
cd "$HALLULENS_DIR"
bash "$SCRIPT"
echo "Finished: $(date)"

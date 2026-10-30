#!/bin/bash
# Run on a login node with network access, not via qsub.
set -euo pipefail
[[ -f scripts/prefetch_hf_assets.py ]] || { echo "Run from repository root" >&2; exit 2; }
source jobs/aspire2a/settings.local.sh
: "${CONDA_ENV:?}"
: "${HF_HOME:?}"
[[ "$CONDA_ENV$HF_HOME" != *REPLACE_WITH* ]] || { echo "Edit settings.local.sh" >&2; exit 2; }
module purge
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
module load "${CONDA_MODULE:-miniforge3/24.3.0}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export HF_HUB_CACHE="$HF_HOME/hub" HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
python scripts/prefetch_hf_assets.py \
  --config configs/benchmark_suite.full.json \
  --config configs/controlled_benchmark.full.json \
  --config configs/detection.controlled.confirmatory.json
python scripts/prepare_benchmark_suite.py --download \
  --benchmark halueval_qa --benchmark ragtruth --benchmark mu_shroom --benchmark truthfulqa

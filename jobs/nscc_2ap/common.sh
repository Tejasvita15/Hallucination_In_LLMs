#!/bin/bash
set -euo pipefail
cd "${PBS_O_WORKDIR:?Submit qsub from the repository root}"
[[ -f scripts/run_halueval_detection.py ]] || { echo "Not the repository root" >&2; exit 2; }
mkdir -p logs/nscc_2ap
exec > >(tee -a "logs/nscc_2ap/${PBS_JOBNAME}_${PBS_JOBID}.log") 2>&1
printf 'Job ID: %s\nJob name: %s\nNode: %s\nStart: %s\nWorkdir: %s\nQueue: %s\n' \
  "$PBS_JOBID" "$PBS_JOBNAME" "$(hostname)" "$(date)" "$PBS_O_WORKDIR" "${PBS_QUEUE:-unknown}"
trap 'rc=$?; echo "Finished: $(date) | Exit code: $rc"' EXIT
[[ -f jobs/nscc_2ap/settings.local.sh ]] && source jobs/nscc_2ap/settings.local.sh
: "${CONDA_ENV:?Set CONDA_ENV in jobs/nscc_2ap/settings.local.sh}"
: "${HF_HOME:?Set HF_HOME to allocated shared storage}"
[[ "$CONDA_ENV$HF_HOME" != *REPLACE_WITH* ]] || { echo "Edit settings.local.sh first" >&2; exit 2; }
module purge
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
module load "${CONDA_MODULE:-miniforge3/24.3.0}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
printf 'Job: %s | Host: %s | Started: %s\n' "$PBS_JOBID" "$(hostname)" "$(date)"
module list 2>&1
printf 'Conda environment: %s\nHF cache: %s\n' "$CONDA_PREFIX" "$HF_HOME"
command -v python
python --version
python - <<'PY'
import sys, torch
print('Python:', sys.executable)
print('PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable: refusing to run GPU job on CPU')
print('GPU:', torch.cuda.get_device_name(0))
x = torch.ones(2, device='cuda')
print('CUDA check:', (x+x).tolist())
PY

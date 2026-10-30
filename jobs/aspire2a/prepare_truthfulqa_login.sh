#!/bin/bash
# Compatibility repair on the login node; no generation or GPU smoke run.
set -euo pipefail
[[ -f scripts/run_truthfulqa_mc.py ]] || { echo 'Run from the repository root' >&2; exit 2; }
source jobs/aspire2a/settings.local.sh
: "${CONDA_ENV:?}" "${HF_HOME:?}"
module purge
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
module load "${CONDA_MODULE:-miniforge3/24.3.0}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export TRUTHFULQA_VENV="${TRUTHFULQA_VENV:-${CONDA_ENV}-truthfulqa-v1}"
python - <<'PY'
import os
from pathlib import Path
base = Path(os.environ['CONDA_PREFIX']).resolve()
target = Path(os.environ['TRUTHFULQA_VENV']).resolve()
if target == base or base in target.parents or target in base.parents:
    raise SystemExit('TRUTHFULQA_VENV must be separate from the active conda environment')
PY
if [[ ! -f "$TRUTHFULQA_VENV/pyvenv.cfg" ]]; then
  python -m venv --system-site-packages "$TRUTHFULQA_VENV"
fi
source "$TRUTHFULQA_VENV/bin/activate"
# Keep the existing CUDA PyTorch build; install overrides only into this venv.
python - <<'PY'
import importlib.metadata, os, sys
from pathlib import Path
if Path(sys.prefix).resolve() != Path(os.environ['TRUTHFULQA_VENV']).resolve():
    raise SystemExit('Refusing to install outside the TruthfulQA venv')
Path(sys.prefix, 'torch-constraint.txt').write_text('torch==' + importlib.metadata.version('torch') + '\n')
PY
python -m pip install -c "$TRUTHFULQA_VENV/torch-constraint.txt" -r requirements-truthfulqa.txt
: "${HF_HOME:?Set HF_HOME before selecting the thesis cache}"
export HF_HOME
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets-thesis-3.6.0"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
python - <<'PY'
import importlib.metadata, json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))
# Import the evaluator, including its model registry, before allocating GPU time.
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager, get_task_dict
from hallu_llm.truthfulqa_tasks import task_configs
tasks = get_task_dict(task_configs(), TaskManager())
print('TruthfulQA task sizes:', {name: len(task.validation_docs()) for name, task in tasks.items()})
versions = {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'tokenizers', 'peft', 'accelerate', 'huggingface-hub', 'datasets', 'lm-eval']}
Path(sys.prefix, 'truthfulqa_ready.json').write_text(json.dumps(versions, indent=2))
print(json.dumps(versions, indent=2))
PY
python -m pip freeze > "$TRUTHFULQA_VENV/environment.lock.txt"
echo 'TruthfulQA preparation complete. Resubmit only the three TruthfulQA PBS jobs.'

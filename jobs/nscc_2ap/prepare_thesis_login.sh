#!/bin/bash
# Cache assets online; no model inference and no smoke experiment.
set -euo pipefail
[[ -f scripts/run_token_confidence.py ]] || { echo 'Run from the repository root' >&2; exit 2; }
source jobs/nscc_2ap/settings.local.sh
: "${CONDA_ENV:?}" "${HF_HOME:?}"
module purge
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
module load "${CONDA_MODULE:-miniforge3/24.3.0}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
source jobs/nscc_2ap/thesis_cache.sh
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
python -m pip install -r requirements-thesis.txt
python scripts/prefetch_hf_assets.py --config configs/benchmark_suite.thesis.json \
  --config configs/detection.review-v3.qwen-en-zh.150.json
python scripts/prepare_benchmark_suite.py --config configs/benchmark_suite.thesis.json --download \
  --benchmark halueval_qa --benchmark ragtruth --benchmark mu_shroom
python - <<'PY'
import importlib.metadata, json, os
from pathlib import Path
from datasets import load_dataset
from lm_eval.tasks import TaskManager, get_task_dict
data = load_dataset('truthfulqa/truthful_qa', 'multiple_choice', split='validation')
# Instantiate task datasets now: GPU nodes run with networking disabled.
get_task_dict(['truthfulqa_mc1', 'truthfulqa_mc2'], TaskManager())
manifest = {'truthfulqa_questions': len(data), 'packages': {
    p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'datasets', 'lm_eval']}}
manifest['datasets_cache'] = os.environ['HF_DATASETS_CACHE']
Path(os.environ['HF_DATASETS_CACHE'], 'thesis_prepared.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
PY
python -m pip freeze > jobs/nscc_2ap/thesis_environment.lock.txt
echo 'Preparation complete. Next: bash jobs/nscc_2ap/submit_thesis_full.sh'

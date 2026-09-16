# Copy to settings.local.sh and edit on NSCC (local settings are Git-ignored).
export NSCC_PROJECT_ID="13014407"
export CONDA_MODULE="miniforge3/24.3.0"
export CONDA_ENV="/scratch/users/ntu/tejasvit/conda_envs/hallucination"
# Shared storage visible on login and GPU nodes; do not use node-local /tmp.
export HF_HOME="/scratch/users/ntu/tejasvit/huggingface"

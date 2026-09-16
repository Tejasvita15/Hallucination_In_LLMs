# Copy to settings.local.sh and edit on NSCC (local settings are Git-ignored).
export NSCC_PROJECT_ID="REPLACE_WITH_PROJECT_ID"
export CONDA_MODULE="miniforge3"
export CONDA_ENV="REPLACE_WITH_CONDA_ENV_NAME_OR_ABSOLUTE_PATH"
# Shared storage visible on login and GPU nodes; do not use node-local /tmp.
export HF_HOME="/REPLACE_WITH_ALLOCATED_STORAGE/huggingface"

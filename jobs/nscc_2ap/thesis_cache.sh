#!/bin/bash
# Source after settings.local.sh. Never delete or reuse the newer datasets cache.
: "${HF_HOME:?Set HF_HOME before selecting the thesis cache}"
export HF_HOME
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets-thesis-3.6.0"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

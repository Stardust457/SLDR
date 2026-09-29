#!/usr/bin/env bash
set -euo pipefail
model=qwen25
model_dir=Qwen2.5_7B_Instruct
model_id="${QWEN25_MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
scaling_script="$model_dir/scaling_qwen.py"
DATASETS="${DATASETS:-sst2}"
source "$(dirname -- "${BASH_SOURCE[0]}")/downstream.sh"

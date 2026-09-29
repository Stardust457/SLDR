#!/usr/bin/env bash
set -euo pipefail
model=llama3
model_dir=Llama3_8B_Instruct
model_id="${LLAMA3_MODEL_ID:-meta-llama/Meta-Llama-3-8B-Instruct}"
scaling_script="$model_dir/scaling_llama.py"
DATASETS="${DATASETS:-sst2}"
source "$(dirname -- "${BASH_SOURCE[0]}")/downstream.sh"

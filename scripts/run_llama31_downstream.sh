#!/usr/bin/env bash
set -euo pipefail
model=llama31
model_dir=Llama_3.1_8B_Instruct
model_id="${LLAMA31_MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
scaling_script="$model_dir/scaling_llama.py"
DATASETS="${DATASETS:-sst2 agnews gsm8k alpaca magicoder}"
source "$(dirname -- "${BASH_SOURCE[0]}")/downstream.sh"

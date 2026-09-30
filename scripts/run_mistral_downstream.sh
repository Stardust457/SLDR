#!/usr/bin/env bash
set -euo pipefail
model=mistral
model_dir=Mistral_7B_Instruct_V0.2
model_id="${MISTRAL_MODEL_ID:-mistralai/Mistral-7B-Instruct-v0.2}"
scaling_script="$model_dir/scaling_mistral.py"
DATASETS="${DATASETS:-sst2}"
source "$(dirname -- "${BASH_SOURCE[0]}")/downstream.sh"

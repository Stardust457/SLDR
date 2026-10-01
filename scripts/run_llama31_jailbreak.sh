#!/usr/bin/env bash
# Evaluate Zulu and IJP using checkpoints from run_llama31_downstream.sh.
set -euo pipefail
model=llama31
model_dir=Llama_3.1_8B_Instruct
model_id="${LLAMA31_MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
DATASETS="${DATASETS:-sst2 agnews gsm8k alpaca magicoder}"
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
read -r -a attacks <<< "${JAILBREAK_ATTACKS:-zulu ijp}"
load_layers

for dataset in "${datasets[@]}"; do
    set_dataset "$dataset"
    for ratio in "${ratios[@]}"; do
        for seed in "${seeds[@]}"; do
            set_paths
            for variant in "${variants[@]}"; do
                set_eval_args
                for attack in "${attacks[@]}"; do
                    run_unless_complete "$harmful_dir/${attack}_${variant}.json" "$dependency_file" \
                        "$log_dir/${attack}_${variant}.log" \
                        "$PYTHON" "$REPO_ROOT/$model_dir/SST2_dataset/eval_on_jailbreak_$attack.py" \
                        "${eval_args[@]}" --max_samples "$HARMFUL_MAX_SAMPLES" \
                        --jailbreak_data_path "$REPO_ROOT/data/benchmarks/${attack}_jailbreak.json" \
                        --output_path "$harmful_dir/${attack}_${variant}.json"
                done
            done
        done
    done
done

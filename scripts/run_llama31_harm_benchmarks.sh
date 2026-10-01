#!/usr/bin/env bash
# Evaluate DirectHarm4, HarmBench and HEx-PHI using existing downstream checkpoints.
set -euo pipefail
model=llama31
model_dir=Llama_3.1_8B_Instruct
model_id="${LLAMA31_MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
DATASETS="${DATASETS:-sst2 agnews gsm8k alpaca magicoder}"
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
load_layers

for dataset in "${datasets[@]}"; do
    set_dataset "$dataset"
    for ratio in "${ratios[@]}"; do
        for seed in "${seeds[@]}"; do
            set_paths
            for variant in "${variants[@]}"; do
                set_eval_args
                other_dir="$harmful_dir/other_${variant}"
                if [[ "$RESUME" == 1 ]] && \
                    is_complete "$other_dir/directHarm4_eval.json" "$dependency_file" && \
                    is_complete "$other_dir/harmbench_eval.json" "$dependency_file" && \
                    is_complete "$other_dir/phi_eval.json" "$dependency_file"; then
                    printf '[SKIP] %s\n' "$other_dir"
                else
                    run_logged "$log_dir/other_${variant}.log" \
                        "$PYTHON" "$REPO_ROOT/$model_dir/SST2_dataset/eval_other_harmfulbench.py" \
                        "${eval_args[@]}" --dataset_dir "$REPO_ROOT/data/benchmarks" \
                        --output_path "$other_dir"
                fi
            done
        done
    done
done

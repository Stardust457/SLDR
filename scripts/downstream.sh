#!/usr/bin/env bash
# Shared downstream workflow, sourced by the four model entry points.
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
STAGES="${STAGES:-select train downstream harmful judge}"

for stage in $STAGES; do
    case "$stage" in
        select|train|downstream|harmful|judge) ;;
        *) printf 'Unknown stage: %s\n' "$stage" >&2; exit 2 ;;
    esac
done
if has_stage judge && [[ " $DATASETS " == *" alpaca "* && "$DRY_RUN" != 1 && -z "${OPENAI_API_KEY:-}" ]]; then
    printf 'Alpaca judge requires OPENAI_API_KEY; use STAGES="select train downstream harmful" to collect answers first.\n' >&2
    exit 2
fi

if has_stage select; then select_layers; fi
if has_stage train || has_stage downstream || has_stage harmful; then load_layers; fi

for dataset in "${datasets[@]}"; do
    set_dataset "$dataset"
    if [[ ! -f "$task_dir/safeguard.py" ]]; then
        printf '[SKIP] No implementation for %s/%s\n' "$model" "$dataset"
        continue
    fi
    for ratio in "${ratios[@]}"; do
        for seed in "${seeds[@]}"; do
            set_paths
            if has_stage train; then
                train_args=(--model_id "$model_id" --cache_dir "$CACHE_DIR"
                    --ft_data_path "$REPO_ROOT/data/downstream/$dataset.json"
                    --harmful_data_path "$REPO_ROOT/data/poisoning/beavertails_harmful_train.json"
                    --safety_data_path "$REPO_ROOT/data/alignment/aligned_data_100.json"
                    --poison_ratio "$ratio" --max_samples "$train_samples" --seed "$seed"
                    --output_path "$ft_dir" --safety_output_path "$safety_dir"
                    --safety_layers "$min_layer" "$max_layer")
                # Stage 1: task LoRA, including the configured poison ratio.
                if [[ "$RESUME" != 1 || ! -f "$ft_dir/.complete" ]]; then
                    if [[ "$DRY_RUN" != 1 ]]; then rm -f -- "$ft_dir/.complete" "$safety_dir/.complete"; fi
                    run_logged "$log_dir/train.log" "$PYTHON" "$task_dir/safeguard.py" \
                        "${train_args[@]}" --skip_defense
                    if [[ "$DRY_RUN" != 1 ]]; then touch "$ft_dir/.complete"; fi
                else
                    printf '[SKIP] Task fine-tuning: %s\n' "$tag"
                fi
                # Stage 2: merge task LoRA, then recover only the selected two layers.
                if [[ "$RESUME" != 1 || ! -f "$safety_dir/.complete" ]]; then
                    if [[ "$DRY_RUN" != 1 ]]; then rm -f -- "$safety_dir/.complete"; fi
                    run_logged "$log_dir/recovery.log" "$PYTHON" "$task_dir/safeguard.py" \
                        "${train_args[@]}" --skip_lora_training
                    if [[ "$DRY_RUN" != 1 ]]; then touch "$safety_dir/.complete"; fi
                else
                    printf '[SKIP] Safety recovery: %s\n' "$tag"
                fi
            fi

            for variant in "${variants[@]}"; do
                if has_stage downstream || has_stage harmful; then set_eval_args; fi
                if has_stage downstream; then
                    run_unless_complete "$downstream_dir/${dataset}_${variant}.json" "$dependency_file" \
                        "$log_dir/downstream_${variant}.log" "$PYTHON" "$task_dir/eval_$dataset.py" \
                        "${eval_args[@]}" --max_samples "$eval_samples" \
                        --output_path "$downstream_dir/${dataset}_${variant}.json"
                fi
                # The main safety benchmark stays with each downstream experiment.
                if has_stage harmful; then
                    run_unless_complete "$harmful_dir/beavertails_${variant}.json" "$dependency_file" \
                        "$log_dir/beavertails_${variant}.log" "$PYTHON" "$task_dir/eval_beavertails.py" \
                        "${eval_args[@]}" --max_samples "$HARMFUL_MAX_SAMPLES" \
                        --output_path "$harmful_dir/beavertails_${variant}.json"
                fi
                if has_stage judge && [[ "$dataset" == alpaca ]]; then
                    judge_args=(--input_file "$downstream_dir/alpaca_${variant}.json"
                        --output_file "$downstream_dir/alpaca_${variant}_judge.json"
                        --model "${GPT_JUDGE_MODEL:-gpt-4o}" --max_workers "${GPT_JUDGE_WORKERS:-4}")
                    if [[ -n "${GPT_JUDGE_BASE_URL:-}" ]]; then judge_args+=(--base_url "$GPT_JUDGE_BASE_URL"); fi
                    run_unless_complete "$downstream_dir/alpaca_${variant}_judge.json.complete" \
                        "$downstream_dir/alpaca_${variant}.json" "$log_dir/alpaca_${variant}_judge.log" \
                        "$PYTHON" "$REPO_ROOT/Llama_3.1_8B_Instruct/Alpaca_dataset/gpt_judge/evaluate_with_gpt4o.py" \
                        "${judge_args[@]}"
                    if [[ "$DRY_RUN" != 1 ]]; then touch "$downstream_dir/alpaca_${variant}_judge.json.complete"; fi
                fi
            done
        done
    done
done

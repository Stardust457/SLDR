#!/usr/bin/env bash
# Shared configuration and small helpers; source from an experiment entry point.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
CACHE_DIR="${CACHE_DIR:-$REPO_ROOT/autodl-tmp}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/results/experiments}"
POISON_RATIOS="${POISON_RATIOS:-0.1}"
SEEDS="${SEEDS:-42}"
VARIANTS="${VARIANTS:-defended}"
SELECT_BATCH_SIZE="${SELECT_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
HARMFUL_MAX_SAMPLES="${HARMFUL_MAX_SAMPLES:-1000}"
AVG_K="${AVG_K:-100}"
HARM_THRESHOLD="${HARM_THRESHOLD:-0.0}"
RESUME="${RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONUNBUFFERED=1

case "${1:-}" in
    --dry-run) DRY_RUN=1 ;;
    --help|-h)
        printf 'Usage: bash %s [--dry-run]\n' "${0##*/}"
        printf 'Configure DATASETS, POISON_RATIOS, TRAIN_SAMPLES, SEEDS, VARIANTS, RUN_DIR. See scripts/README.md.\n'
        exit 0 ;;
    "") ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
esac

read -r -a datasets <<< "$DATASETS"
read -r -a ratios <<< "$POISON_RATIOS"
read -r -a seeds <<< "$SEEDS"
read -r -a variants <<< "$VARIANTS"
layer_file="$RUN_DIR/layers/$model.json"

has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

run_logged() {
    local log_file="$1"
    shift
    printf '\n[RUN]'
    printf ' %q' "$@"
    printf '\n'
    if [[ "$DRY_RUN" == 1 ]]; then return 0; fi
    mkdir -p -- "$(dirname -- "$log_file")"
    "$@" 2>&1 | tee "$log_file"
}

is_complete() {
    [[ -f "$1" ]] && { [[ -z "$2" ]] || [[ "$1" -nt "$2" ]]; }
}

run_unless_complete() {
    local completion_file="$1" dependency_file="$2"
    shift 2
    if [[ "$RESUME" == 1 ]] && is_complete "$completion_file" "$dependency_file"; then
        printf '[SKIP] %s\n' "$completion_file"
    else
        run_logged "$@"
    fi
}

select_layers() {
    # A completed diagnosis is shared by all tasks, ratios and training sizes.
    if [[ -f "$layer_file" ]]; then
        printf '[SKIP] Layer selection: %s\n' "$layer_file"
        return 0
    fi
    local alphas
    read -r -a alphas <<< "${ALPHAS:-0.1 0.2}"
    run_logged "$RUN_DIR/logs/$model/select.log" "$PYTHON" "$scaling_script" \
        --model_id "$model_id" --cache_dir "$CACHE_DIR" \
        --data_path "$REPO_ROOT/data/diagnostics/overrejection_final.json" \
        --alphas "${alphas[@]}" --start_layer "${START_LAYER:-0}" \
        --batch_size "$SELECT_BATCH_SIZE" --output_path "$layer_file"
}

load_layers() {
    if [[ "$DRY_RUN" == 1 && ! -f "$layer_file" ]]; then
        min_layer='<l_min>'; max_layer='<l_max>'; route_layer='<l_max+1>'
        return 0
    fi
    local selection
    selection="$("$PYTHON" - "$layer_file" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"Missing layer selection: {path}. Run the downstream script with STAGES=select first.")
layers = json.loads(path.read_text(encoding="utf-8"))
l_min, l_max = int(layers["unsafe_layer"]), int(layers["safety_layer"])
if l_min == l_max:
    raise SystemExit("Layer diagnosis did not identify two distinct endpoints.")
# LoRA uses block indices; hidden_states[0] is embeddings.
print(l_min, l_max, l_max + 1)
PY
    )"
    read -r min_layer max_layer route_layer <<< "$selection"
}

set_dataset() {
    default_train_samples=1000
    case "$1" in
        sst2) dataset_dir=SST2_dataset; eval_samples=872 ;;
        agnews) dataset_dir=AGNews_dataset; eval_samples=1000 ;;
        gsm8k) dataset_dir=GSM8K_dataset; eval_samples=1000 ;;
        alpaca) dataset_dir=Alpaca_dataset; default_train_samples=700; eval_samples=104 ;;
        magicoder) dataset_dir=Magicoder_dataset; eval_samples=164 ;;
        *) printf 'Unknown dataset: %s\n' "$1" >&2; exit 2 ;;
    esac
    train_samples="${TRAIN_SAMPLES:-$default_train_samples}"
    task_dir="$model_dir/$dataset_dir"
}

set_paths() {
    tag="$model/$dataset/p$ratio"
    # Keep existing default checkpoints usable; separate custom training sizes.
    if [[ "$train_samples" != "$default_train_samples" ]]; then tag+="/n$train_samples"; fi
    tag+="/seed$seed"
    ft_dir="$RUN_DIR/checkpoints/$tag/ft_lora"
    safety_dir="$RUN_DIR/checkpoints/$tag/safety_lora"
    log_dir="$RUN_DIR/logs/$tag"
    downstream_dir="$RUN_DIR/downstream/$tag"
    harmful_dir="$RUN_DIR/harmful/$tag"
}

set_eval_args() {
    eval_args=(--model_id "$model_id" --cache_dir "$CACHE_DIR"
        --ft_lora_path "$ft_dir" --safety_lora_path "$safety_dir"
        --unsafe_data_path "$REPO_ROOT/data/routing/harmful_prompt.json"
        --safe_data_path "$REPO_ROOT/data/routing/safe_prompt.json"
        --layer_num "$route_layer" --avg_k "$AVG_K"
        --harm_threshold "$HARM_THRESHOLD" --batch_size "$EVAL_BATCH_SIZE")
    case "$variant" in
        defended) dependency_file="$safety_dir/.complete" ;;
        sft) dependency_file="$ft_dir/.complete"; eval_args+=(--skip_defense) ;;
        base) dependency_file=""; eval_args+=(--skip_finetune --skip_defense) ;;
        *) printf 'Unknown variant: %s\n' "$variant" >&2; exit 2 ;;
    esac
}

#!/usr/bin/env bash
# Run independently after response collection to free the GPU for Llama Guard.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/results/experiments}"
CACHE_DIR="${CACHE_DIR:-$REPO_ROOT/autodl-tmp}"
GUARD_MODEL="${GUARD_MODEL:-meta-llama/Llama-Guard-3-8B}"
GUARD_SERVED_MODEL="${GUARD_SERVED_MODEL:-llama-guard-3-8b}"
GUARD_PORT="${GUARD_PORT:-8612}"
GUARD_URL="${LLAMA_GUARD_URL:-http://127.0.0.1:$GUARD_PORT/v1}"
GUARD_INPUT_DIR="${GUARD_INPUT_DIR:-$RUN_DIR/harmful}"
GUARD_OUTPUT_DIR="${GUARD_OUTPUT_DIR:-$RUN_DIR/llama_guard}"
GUARD_WORKERS="${GUARD_WORKERS:-40}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

command="${1:-score}"
if [[ "$#" -gt 0 ]]; then shift; fi
case "$command" in
    serve)
        args=("$PYTHON" -m vllm.entrypoints.openai.api_server
            --model "$GUARD_MODEL" --served-model-name "$GUARD_SERVED_MODEL"
            --download-dir "$CACHE_DIR" --host 127.0.0.1 --port "$GUARD_PORT") ;;
    score)
        args=("$PYTHON" "$REPO_ROOT/llama_guard.py"
            --input_dir "$GUARD_INPUT_DIR" --output_dir "$GUARD_OUTPUT_DIR"
            --base_url "$GUARD_URL" --model "$GUARD_SERVED_MODEL"
            --max_workers "$GUARD_WORKERS") ;;
    --help|-h)
        cat <<'HELP'
Usage:
  bash scripts/run_llama_guard.sh serve  # foreground vLLM service
  bash scripts/run_llama_guard.sh score  # run in another terminal

Configure RUN_DIR, GUARD_MODEL (local path or HF ID), GUARD_PORT,
GUARD_SERVED_MODEL, LLAMA_GUARD_URL, GUARD_WORKERS, GUARD_INPUT_DIR,
GUARD_OUTPUT_DIR, and LLAMA_GUARD_API_KEY using environment variables.
Extra arguments are passed to vLLM or llama_guard.py. DRY_RUN=1 prints only.
Raw responses are preserved; scored copies and summary.json go to GUARD_OUTPUT_DIR.
HELP
        exit 0 ;;
    *) printf 'Unknown command: %s\n' "$command" >&2; exit 2 ;;
esac
args+=("$@")
printf '[RUN]'
printf ' %q' "${args[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" == 1 ]]; then exit 0; fi
exec "${args[@]}"

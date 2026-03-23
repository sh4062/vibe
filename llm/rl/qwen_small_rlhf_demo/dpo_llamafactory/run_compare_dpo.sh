#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
MODEL_PATH="${MODEL_PATH:-/Users/a1-6/Documents/code/vibe/Qwen3.5-0.8B}"
ADAPTER_PATH="${ADAPTER_PATH:-$SCRIPT_DIR/outputs/dpo_lora}"
COMPARE_OUTPUT="${COMPARE_OUTPUT:-$SCRIPT_DIR/outputs/dpo_compare.jsonl}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
LIMIT_SAMPLES="${LIMIT_SAMPLES:-10}"

export MODEL_PATH ADAPTER_PATH COMPARE_OUTPUT MAX_NEW_TOKENS LIMIT_SAMPLES

python3 "$SCRIPT_DIR/compare_dpo.py"

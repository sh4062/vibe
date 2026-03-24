#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
unset INSPECT_OUTPUT
unset INSPECT_SPLIT
unset MAX_NEW_TOKENS

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
export ADAPTER_PATH="${ADAPTER_PATH:-$SCRIPT_DIR/outputs/grpo_numina_lora}"
export INSPECT_OUTPUT="$SCRIPT_DIR/outputs/inspect_generations.jsonl"
export INSPECT_SPLIT="test[:5]"
export MAX_NEW_TOKENS=256

python3 "$SCRIPT_DIR/inspect_generations.py"

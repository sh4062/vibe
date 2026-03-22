#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-$SCRIPT_DIR/outputs/ppo_lora}"
COMPARE_OUTPUT="${COMPARE_OUTPUT:-$SCRIPT_DIR/outputs/ppo_compare.jsonl}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-48}"

export MODEL_PATH ADAPTER_PATH COMPARE_OUTPUT MAX_NEW_TOKENS

python3 "$SCRIPT_DIR/compare_ppo.py"

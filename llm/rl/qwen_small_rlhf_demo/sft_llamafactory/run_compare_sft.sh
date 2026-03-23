#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-$SCRIPT_DIR/outputs/sft_lora}"
COMPARE_OUTPUT="${COMPARE_OUTPUT:-$SCRIPT_DIR/outputs/sft_compare.jsonl}"

export MODEL_PATH ADAPTER_PATH COMPARE_OUTPUT

python3 "$SCRIPT_DIR/compare_sft.py"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
unset ADAPTER_PATH
unset COMPARE_OUTPUT
unset LIMIT_SAMPLES
unset SAMPLE_COUNT

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
export ADAPTER_PATH="$SCRIPT_DIR/outputs/grpo_lora"
export COMPARE_OUTPUT="$SCRIPT_DIR/outputs/grpo_compare.jsonl"
export LIMIT_SAMPLES=10
export SAMPLE_COUNT=3

python3 "$SCRIPT_DIR/compare_grpo.py"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
unset OUTPUT_DIR
unset DATASET_ID
unset TRAIN_SPLIT
unset EVAL_SPLIT

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
export OUTPUT_DIR="$SCRIPT_DIR/outputs/grpo_numina_lora"
export DATASET_ID="AI-MO/NuminaMath-TIR"
export TRAIN_SPLIT="train[:1%]"
export EVAL_SPLIT="test[:1%]"

python3 "$SCRIPT_DIR/train_grpo_numina.py"

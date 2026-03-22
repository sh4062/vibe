#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/outputs/ppo_lora}"

export MODEL_PATH OUTPUT_DIR
export TRL_EXPERIMENTAL_SILENCE=1

python3 "$SCRIPT_DIR/train_ppo.py"

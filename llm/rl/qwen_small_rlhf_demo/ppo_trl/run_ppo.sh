#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
MODEL_PATH="${MODEL_PATH:-/Users/a1-6/Documents/code/vibe/Qwen3.5-0.8B}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/outputs/ppo_lora}"

export MODEL_PATH OUTPUT_DIR

python3 "$SCRIPT_DIR/train_ppo.py"

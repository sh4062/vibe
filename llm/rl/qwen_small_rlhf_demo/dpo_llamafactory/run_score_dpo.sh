#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
MODEL_PATH="${MODEL_PATH:-/Users/a1-6/Documents/code/vibe/Qwen3.5-0.8B}"
ADAPTER_PATH="${ADAPTER_PATH:-$SCRIPT_DIR/outputs/dpo_lora}"
SCORE_OUTPUT="${SCORE_OUTPUT:-$SCRIPT_DIR/outputs/dpo_score.json}"

export MODEL_PATH ADAPTER_PATH SCORE_OUTPUT

python3 "$SCRIPT_DIR/score_dpo.py"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
unset ADAPTER_PATH
unset SCORE_OUTPUT
unset LIMIT_SAMPLES

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
export ADAPTER_PATH="$SCRIPT_DIR/outputs/reward_model_lora"
export SCORE_OUTPUT="$SCRIPT_DIR/outputs/reward_model_score.json"
export LIMIT_SAMPLES=10

python3 "$SCRIPT_DIR/score_reward_model.py"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
unset OUTPUT_DIR
unset DATASET_ID
unset TRAIN_SPLIT
unset EVAL_SPLIT
unset RESUME_FROM_CHECKPOINT

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
export OUTPUT_DIR="$SCRIPT_DIR/outputs/grpo_numina_lora"
export DATASET_ID="AI-MO/NuminaMath-TIR"
export TRAIN_SPLIT="train[:5%]"
export EVAL_SPLIT="test[:5%]"

if [[ -n "${GRPO_RESUME_FROM:-}" ]]; then
  if [[ -f "$GRPO_RESUME_FROM/trainer_state.json" ]]; then
    export RESUME_FROM_CHECKPOINT="$GRPO_RESUME_FROM"
  elif [[ -d "$GRPO_RESUME_FROM" ]]; then
    latest_checkpoint=$(find "$GRPO_RESUME_FROM" -maxdepth 1 -type d -name 'checkpoint-*' | while read -r d; do [[ -f "$d/trainer_state.json" ]] && echo "$d"; done | sort -V | tail -n 1 || true)
    if [[ -n "$latest_checkpoint" ]]; then
      export RESUME_FROM_CHECKPOINT="$latest_checkpoint"
    fi
  fi
fi

python3 "$SCRIPT_DIR/train_grpo_numina.py"

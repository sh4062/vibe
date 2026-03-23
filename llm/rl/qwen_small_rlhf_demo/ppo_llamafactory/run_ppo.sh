#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
MODEL_PATH="${MODEL_PATH:-/Users/a1-6/Documents/code/vibe/Qwen3.5-0.8B}"
TEMPLATE="${TEMPLATE:-default}"
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-$SCRIPT_DIR/outputs/reward_model_lora}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/outputs/ppo_lora}"

mkdir -p "$SCRIPT_DIR/outputs"

llamafactory-cli train \
  --stage ppo \
  --do_train true \
  --model_name_or_path "$MODEL_PATH" \
  --dataset_dir "$SCRIPT_DIR/data" \
  --dataset tiny_ppo_prompts_local \
  --template "$TEMPLATE" \
  --finetuning_type lora \
  --lora_target all \
  --quantization_bit 4 \
  --cutoff_len 256 \
  --max_samples 32 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --num_train_epochs 4.0 \
  --logging_steps 1 \
  --save_steps 20 \
  --warmup_steps 0 \
  --report_to none \
  --output_dir "$OUTPUT_DIR" \
  --overwrite_output_dir true \
  --reward_model "$REWARD_MODEL_PATH" \
  --reward_model_type lora \
  --top_k 0 \
  --top_p 0.9 \
  --plot_loss true

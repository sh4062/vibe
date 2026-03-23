#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
TEMPLATE="${TEMPLATE:-qwen}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/outputs/sft_lora}"

mkdir -p "$SCRIPT_DIR/outputs"

llamafactory-cli train \
  --stage sft \
  --do_train true \
  --model_name_or_path "$MODEL_PATH" \
  --dataset_dir "$SCRIPT_DIR/data" \
  --dataset tiny_sft_local \
  --template "$TEMPLATE" \
  --finetuning_type lora \
  --lora_target all \
  --quantization_bit 4 \
  --cutoff_len 256 \
  --max_samples 100 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --num_train_epochs 10.0 \
  --logging_steps 1 \
  --save_steps 20 \
  --warmup_steps 0 \
  --report_to none \
  --output_dir "$OUTPUT_DIR" \
  --overwrite_output_dir true \
  --plot_loss true

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
TEMPLATE="${TEMPLATE:-qwen}"
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-$SCRIPT_DIR/outputs/reward_model_lora}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/outputs/ppo_lora}"
PPO_RESUME_FROM="${PPO_RESUME_FROM:-}"

mkdir -p "$SCRIPT_DIR/outputs"

init_adapter=""
if [[ -n "$PPO_RESUME_FROM" ]]; then
  if [[ -f "$PPO_RESUME_FROM/adapter_config.json" ]]; then
    init_adapter="$PPO_RESUME_FROM"
  elif [[ -d "$PPO_RESUME_FROM" ]]; then
    latest_checkpoint=$(find "$PPO_RESUME_FROM" -maxdepth 1 -type d -name 'checkpoint-*' | while read -r d; do [[ -f "$d/adapter_config.json" ]] && echo "$d"; done | sort -V | tail -n 1 || true)
    if [[ -n "$latest_checkpoint" ]]; then
      init_adapter="$latest_checkpoint"
    fi
  fi
fi

TRAIN_ARGS=(
  train
  --stage ppo
  --do_train true
  --model_name_or_path "$MODEL_PATH"
  --dataset_dir "$SCRIPT_DIR/data"
  --dataset tiny_ppo_prompts_local
  --template "$TEMPLATE"
  --finetuning_type lora
  --lora_target all
  --quantization_bit 4
  --cutoff_len 256
  --max_samples 500
  --per_device_train_batch_size 1
  --gradient_accumulation_steps 4
  --learning_rate 5e-5
  --num_train_epochs 4.0
  --logging_steps 1
  --save_steps 20
  --warmup_steps 0
  --report_to none
  --output_dir "$OUTPUT_DIR"
  --overwrite_output_dir true
  --reward_model "$REWARD_MODEL_PATH"
  --reward_model_type lora
  --top_k 0
  --top_p 0.9
  --plot_loss true
)

if [[ -n "$init_adapter" ]]; then
  echo "Initializing PPO from existing adapter: $init_adapter"
  TRAIN_ARGS+=(--adapter_name_or_path "$init_adapter")
fi

llamafactory-cli "${TRAIN_ARGS[@]}"

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not $env:MODEL_PATH) {
    $env:MODEL_PATH = "E:\code\vibe\Qwen3.5-0.8B"
}

if (-not $env:TEMPLATE) {
    $env:TEMPLATE = "default"
}

if (-not $env:REWARD_MODEL_PATH) {
    $env:REWARD_MODEL_PATH = Join-Path $ScriptDir "outputs\reward_model_lora"
}

if (-not $env:OUTPUT_DIR) {
    $env:OUTPUT_DIR = Join-Path $ScriptDir "outputs\ppo_lora"
}

New-Item -ItemType Directory -Force -Path (Join-Path $ScriptDir "outputs") | Out-Null

llamafactory-cli train `
  --stage ppo `
  --do_train true `
  --model_name_or_path $env:MODEL_PATH `
  --dataset_dir (Join-Path $ScriptDir "data") `
  --dataset tiny_ppo_prompts_local `
  --template $env:TEMPLATE `
  --finetuning_type lora `
  --lora_target all `
  --quantization_bit 4 `
  --cutoff_len 256 `
  --max_samples 8 `
  --per_device_train_batch_size 1 `
  --gradient_accumulation_steps 4 `
  --learning_rate 1e-5 `
  --num_train_epochs 1.0 `
  --logging_steps 1 `
  --save_steps 20 `
  --warmup_steps 0 `
  --report_to none `
  --output_dir $env:OUTPUT_DIR `
  --overwrite_output_dir true `
  --reward_model $env:REWARD_MODEL_PATH `
  --reward_model_type lora `
  --top_k 0 `
  --top_p 0.9 `
  --plot_loss true

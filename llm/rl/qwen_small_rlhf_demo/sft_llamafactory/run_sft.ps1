$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

@("TEMPLATE", "OUTPUT_DIR") | ForEach-Object {
    Remove-Item "Env:$_" -ErrorAction SilentlyContinue
}

if (-not $env:MODEL_PATH) {
    $env:MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
}

$env:TEMPLATE = "qwen"
$env:OUTPUT_DIR = Join-Path $ScriptDir "outputs\sft_lora"

New-Item -ItemType Directory -Force -Path (Join-Path $ScriptDir "outputs") | Out-Null

llamafactory-cli train `
  --stage sft `
  --do_train true `
  --model_name_or_path $env:MODEL_PATH `
  --dataset_dir (Join-Path $ScriptDir "data") `
  --dataset tiny_sft_local `
  --template $env:TEMPLATE `
  --finetuning_type lora `
  --lora_target all `
  --quantization_bit 4 `
  --cutoff_len 256 `
  --max_samples 100 `
  --per_device_train_batch_size 1 `
  --gradient_accumulation_steps 4 `
  --learning_rate 1e-4 `
  --num_train_epochs 10.0 `
  --logging_steps 1 `
  --save_steps 20 `
  --warmup_steps 0 `
  --report_to none `
  --output_dir $env:OUTPUT_DIR `
  --overwrite_output_dir true `
  --plot_loss true

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

@("OUTPUT_DIR", "DATASET_ID", "TRAIN_SPLIT", "EVAL_SPLIT") | ForEach-Object {
    Remove-Item "Env:$_" -ErrorAction SilentlyContinue
}

if (-not $env:MODEL_PATH) {
    $env:MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
}

$env:OUTPUT_DIR = Join-Path $ScriptDir "outputs\grpo_numina_lora"
$env:DATASET_ID = "AI-MO/NuminaMath-TIR"
$env:TRAIN_SPLIT = "train[:1%]"
$env:EVAL_SPLIT = "test[:1%]"

python (Join-Path $ScriptDir "train_grpo_numina.py")

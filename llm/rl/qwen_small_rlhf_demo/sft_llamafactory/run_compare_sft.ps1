$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not $env:MODEL_PATH) {
    $env:MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
}

if (-not $env:ADAPTER_PATH) {
    $env:ADAPTER_PATH = Join-Path $ScriptDir "outputs\sft_lora"
}

$env:COMPARE_OUTPUT = Join-Path $ScriptDir "outputs\sft_compare.jsonl"

if (-not $env:LIMIT_SAMPLES) {
    $env:LIMIT_SAMPLES = "10"
}

python (Join-Path $ScriptDir "compare_sft.py")

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not $env:MODEL_PATH) {
    $env:MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
}

if (-not $env:ADAPTER_PATH) {
    $env:ADAPTER_PATH = Join-Path $ScriptDir "outputs\dpo_lora"
}

if (-not $env:COMPARE_OUTPUT) {
    $env:COMPARE_OUTPUT = Join-Path $ScriptDir "outputs\dpo_compare.jsonl"
}

if (-not $env:MAX_NEW_TOKENS) {
    $env:MAX_NEW_TOKENS = "64"
}

python (Join-Path $ScriptDir "compare_dpo.py")

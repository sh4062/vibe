$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not $env:MODEL_PATH) {
    $env:MODEL_PATH = "E:\code\vibe\Qwen3.5-0.8B"
}

if (-not $env:ADAPTER_PATH) {
    $env:ADAPTER_PATH = Join-Path $ScriptDir "outputs\dpo_lora"
}

if (-not $env:SCORE_OUTPUT) {
    $env:SCORE_OUTPUT = Join-Path $ScriptDir "outputs\dpo_score.json"
}

python (Join-Path $ScriptDir "score_dpo.py")

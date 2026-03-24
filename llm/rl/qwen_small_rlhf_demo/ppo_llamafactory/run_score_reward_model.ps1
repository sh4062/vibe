$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

@("ADAPTER_PATH", "SCORE_OUTPUT", "LIMIT_SAMPLES") | ForEach-Object {
    Remove-Item "Env:$_" -ErrorAction SilentlyContinue
}

if (-not $env:MODEL_PATH) {
    $env:MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
}

$env:ADAPTER_PATH = Join-Path $ScriptDir "outputs\reward_model_lora"
$env:SCORE_OUTPUT = Join-Path $ScriptDir "outputs\reward_model_score.json"
$env:LIMIT_SAMPLES = "10"

python (Join-Path $ScriptDir "score_reward_model.py")

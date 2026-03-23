$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

@("ADAPTER_PATH", "SCORE_OUTPUT") | ForEach-Object {
    Remove-Item "Env:$_" -ErrorAction SilentlyContinue
}

if (-not $env:MODEL_PATH) {
    $env:MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
}

$env:ADAPTER_PATH = Join-Path $ScriptDir "outputs\dpo_lora"
$env:SCORE_OUTPUT = Join-Path $ScriptDir "outputs\dpo_score.json"

python (Join-Path $ScriptDir "score_dpo.py")

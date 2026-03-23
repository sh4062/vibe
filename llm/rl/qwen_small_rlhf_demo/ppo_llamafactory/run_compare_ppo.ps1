$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

@("ADAPTER_PATH", "COMPARE_OUTPUT", "MAX_NEW_TOKENS", "SAMPLE_COUNT", "LIMIT_SAMPLES") | ForEach-Object {
    Remove-Item "Env:$_" -ErrorAction SilentlyContinue
}

if (-not $env:MODEL_PATH) {
    $env:MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
}

$env:ADAPTER_PATH = Join-Path $ScriptDir "outputs\ppo_lora"
$env:COMPARE_OUTPUT = Join-Path $ScriptDir "outputs\ppo_compare.jsonl"
$env:MAX_NEW_TOKENS = "48"
$env:SAMPLE_COUNT = "3"
$env:LIMIT_SAMPLES = "10"

python (Join-Path $ScriptDir "compare_ppo.py")

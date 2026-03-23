$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

@("ADAPTER_PATH", "COMPARE_OUTPUT", "LIMIT_SAMPLES", "SAMPLE_COUNT") | ForEach-Object {
    Remove-Item "Env:$_" -ErrorAction SilentlyContinue
}

if (-not $env:MODEL_PATH) {
    $env:MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
}

$env:ADAPTER_PATH = Join-Path $ScriptDir "outputs\grpo_lora"
$env:COMPARE_OUTPUT = Join-Path $ScriptDir "outputs\grpo_compare.jsonl"
$env:LIMIT_SAMPLES = "10"
$env:SAMPLE_COUNT = "3"

python (Join-Path $ScriptDir "compare_grpo.py")

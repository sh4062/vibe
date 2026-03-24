$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

@("ADAPTER_PATH", "INSPECT_OUTPUT", "INSPECT_SPLIT", "MAX_NEW_TOKENS") | ForEach-Object {
    Remove-Item "Env:$_" -ErrorAction SilentlyContinue
}

if (-not $env:MODEL_PATH) {
    $env:MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
}

$env:ADAPTER_PATH = Join-Path $ScriptDir "outputs\grpo_numina_lora\checkpoint-290"
$env:INSPECT_OUTPUT = Join-Path $ScriptDir "outputs\inspect_generations.jsonl"
$env:INSPECT_SPLIT = "test[:3]"
$env:MAX_NEW_TOKENS = "512"

python (Join-Path $ScriptDir "inspect_generations.py")

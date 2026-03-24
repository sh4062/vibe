$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

@("INSPECT_OUTPUT", "INSPECT_SPLIT", "MAX_NEW_TOKENS") | ForEach-Object {
    Remove-Item "Env:$_" -ErrorAction SilentlyContinue
}

if (-not $env:MODEL_PATH) {
    $env:MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
}

<<<<<<< HEAD
$env:ADAPTER_PATH = Join-Path $ScriptDir "outputs\grpo_numina_lora\checkpoint-290"
=======
if (-not $env:ADAPTER_PATH) {
    $env:ADAPTER_PATH = Join-Path $ScriptDir "outputs\grpo_numina_lora"
}
>>>>>>> 0a40da4afe7bcb01c2936ea654d967925c0a8303
$env:INSPECT_OUTPUT = Join-Path $ScriptDir "outputs\inspect_generations.jsonl"
$env:INSPECT_SPLIT = "test[:3]"
$env:MAX_NEW_TOKENS = "512"

python (Join-Path $ScriptDir "inspect_generations.py")

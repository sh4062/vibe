$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

@("OUTPUT_DIR", "TRL_EXPERIMENTAL_SILENCE") | ForEach-Object {
    Remove-Item "Env:$_" -ErrorAction SilentlyContinue
}

if (-not $env:MODEL_PATH) {
    $env:MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
}

$env:OUTPUT_DIR = Join-Path $ScriptDir "outputs\ppo_lora"
$env:TRL_EXPERIMENTAL_SILENCE = "1"

python (Join-Path $ScriptDir "train_ppo.py")

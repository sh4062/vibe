$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not $env:MODEL_PATH) {
    $env:MODEL_PATH = "E:\code\vibe\Qwen3.5-0.8B"
}

python $ScriptDir\train_grpo.py

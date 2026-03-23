$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

@("OUTPUT_DIR") | ForEach-Object {
    Remove-Item "Env:$_" -ErrorAction SilentlyContinue
}

if (-not $env:MODEL_PATH) {
    $env:MODEL_PATH = "E:\code\vibe\Qwen2.5-0.5B-Instruct"
}

python $ScriptDir\train_grpo.py

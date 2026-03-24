$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

@("OUTPUT_DIR", "DATASET_ID", "TRAIN_SPLIT", "EVAL_SPLIT", "RESUME_FROM_CHECKPOINT") | ForEach-Object {
    Remove-Item "Env:$_" -ErrorAction SilentlyContinue
}

if (-not $env:MODEL_PATH) {
    $env:MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
}

$env:OUTPUT_DIR = Join-Path $ScriptDir "outputs\grpo_numina_lora"
$env:DATASET_ID = "AI-MO/NuminaMath-TIR"
$env:TRAIN_SPLIT = "train[:5%]"
$env:EVAL_SPLIT = "test[:5%]"

if ($env:GRPO_RESUME_FROM) {
    if ((Test-Path $env:GRPO_RESUME_FROM) -and (Test-Path (Join-Path $env:GRPO_RESUME_FROM "trainer_state.json"))) {
        $env:RESUME_FROM_CHECKPOINT = $env:GRPO_RESUME_FROM
    }
    elseif (Test-Path $env:GRPO_RESUME_FROM) {
        $LatestCheckpoint = Get-ChildItem -Path $env:GRPO_RESUME_FROM -Directory -Filter "checkpoint-*" |
            Where-Object { Test-Path (Join-Path $_.FullName "trainer_state.json") } |
            Sort-Object {
                if ($_.Name -match '^checkpoint-(\d+)$') { [int]$matches[1] } else { -1 }
            } -Descending |
            Select-Object -First 1
        if ($LatestCheckpoint) {
            $env:RESUME_FROM_CHECKPOINT = $LatestCheckpoint.FullName
        }
    }
}

python (Join-Path $ScriptDir "train_grpo_numina.py")

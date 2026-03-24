$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

@("TEMPLATE", "REWARD_MODEL_PATH", "OUTPUT_DIR", "RESUME_CHECKPOINT") | ForEach-Object {
    Remove-Item "Env:$_" -ErrorAction SilentlyContinue
}

if (-not $env:MODEL_PATH) {
    $env:MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
}

$env:TEMPLATE = "qwen"
$env:REWARD_MODEL_PATH = Join-Path $ScriptDir "outputs\reward_model_lora"
$env:OUTPUT_DIR = Join-Path $ScriptDir "outputs\ppo_lora"

New-Item -ItemType Directory -Force -Path (Join-Path $ScriptDir "outputs") | Out-Null

$ResumeFrom = $null
if ($env:PPO_RESUME_FROM) {
    if ((Test-Path $env:PPO_RESUME_FROM) -and (Test-Path (Join-Path $env:PPO_RESUME_FROM "trainer_state.json"))) {
        $ResumeFrom = $env:PPO_RESUME_FROM
    }
    elseif (Test-Path $env:PPO_RESUME_FROM) {
        $LatestCheckpoint = Get-ChildItem -Path $env:PPO_RESUME_FROM -Directory -Filter "checkpoint-*" |
            Sort-Object {
                if ($_.Name -match '^checkpoint-(\d+)$') { [int]$matches[1] } else { -1 }
            } -Descending |
            Select-Object -First 1
        if ($LatestCheckpoint) {
            $ResumeFrom = $LatestCheckpoint.FullName
        }
    }
}

$TrainArgs = @(
  "train",
  "--stage", "ppo",
  "--do_train", "true",
  "--model_name_or_path", $env:MODEL_PATH,
  "--dataset_dir", (Join-Path $ScriptDir "data"),
  "--dataset", "tiny_ppo_prompts_local",
  "--template", $env:TEMPLATE,
  "--finetuning_type", "lora",
  "--lora_target", "all",
  "--quantization_bit", "4",
  "--cutoff_len", "256",
  "--max_samples", "500",
  "--per_device_train_batch_size", "1",
  "--gradient_accumulation_steps", "4",
  "--learning_rate", "5e-5",
  "--num_train_epochs", "4.0",
  "--logging_steps", "1",
  "--save_steps", "20",
  "--warmup_steps", "0",
  "--report_to", "none",
  "--output_dir", $env:OUTPUT_DIR,
  "--reward_model", $env:REWARD_MODEL_PATH,
  "--reward_model_type", "lora",
  "--top_k", "0",
  "--top_p", "0.9",
  "--plot_loss", "true"
)

if ($ResumeFrom) {
    Write-Host "Resuming PPO from checkpoint: $ResumeFrom"
    $TrainArgs += @("--resume_from_checkpoint", $ResumeFrom)
} else {
    $TrainArgs += @("--overwrite_output_dir", "true")
}

llamafactory-cli @TrainArgs

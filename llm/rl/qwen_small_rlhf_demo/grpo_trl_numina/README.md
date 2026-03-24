# GRPO with TRL on NuminaMath-TIR

This folder follows the Hugging Face GRPO cookbook more closely than the toy GRPO demo.

Default choices:

- dataset: `AI-MO/NuminaMath-TIR`
- model: `Qwen/Qwen2.5-0.5B-Instruct`
- rewards:
  - format reward for `<think>...</think><answer>...</answer>`
  - accuracy reward for the final answer
- train split: `train[:5%]`
- eval split: `test[:5%]`
- checkpoints: every `10` steps

## Notes

- The dataset is downloaded at runtime through `datasets.load_dataset(...)`
- This path is better for showing GRPO-style reasoning rewards than the earlier refusal-style toy demo
- A true supervised-style `valid loss` is not always available in GRPO, but this script will save any logged:
  - training loss
  - training reward
  - eval reward
  - per-reward metrics if present

## PowerShell

```powershell
cd E:\code\vibe\llm\rl\qwen_small_rlhf_demo\grpo_trl_numina
$env:MODEL_PATH="E:\code\vibe\Qwen2.5-0.5B-Instruct"
.\run_grpo.ps1
```

## Bash

```bash
cd llm/rl/qwen_small_rlhf_demo/grpo_trl_numina
export MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
bash run_grpo.sh
```

The main training script will save:

- `trainer_log.jsonl`
- `training_loss.png`
- `training_reward.png`
- `eval_reward.png` if eval logs are present
- reward-specific plots when those metrics are logged

During training, these plots are refreshed every `10` steps so you can inspect progress without waiting for the full run to finish.

## Resume training

You can continue from a specific checkpoint directory:

```powershell
$env:GRPO_RESUME_FROM="E:\code\vibe\llm\rl\qwen_small_rlhf_demo\grpo_trl_numina\outputs\grpo_numina_lora\checkpoint-10"
.\run_grpo.ps1
```

Or point to the output root and the launcher will pick the latest `checkpoint-*` automatically:

```powershell
$env:GRPO_RESUME_FROM="E:\code\vibe\llm\rl\qwen_small_rlhf_demo\grpo_trl_numina\outputs\grpo_numina_lora"
.\run_grpo.ps1
```

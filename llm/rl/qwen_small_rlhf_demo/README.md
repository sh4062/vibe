# Qwen Small RLHF Demo

This folder contains six minimal demos for a locally downloaded Qwen small model:

- `sft_llamafactory/`: SFT with LLaMA-Factory
- `dpo_llamafactory/`: DPO with LLaMA-Factory
- `ppo_llamafactory/`: Reward modeling + PPO with LLaMA-Factory
- `ppo_trl/`: PPO with TRL and a rule-based reward
- `grpo_trl/`: GRPO with TRL
- `grpo_trl_numina/`: GRPO with TRL on `AI-MO/NuminaMath-TIR`
- `grpo_verl/`: GRPO with verl

These demos are intentionally tiny so they are easier to run on limited hardware.

`grpo_verl/` is the most infrastructure-heavy option in this folder. It is best treated as a separate Linux/CUDA experiment rather than a drop-in replacement for the lighter LLaMA-Factory demos.

## Windows usage

PowerShell launchers are included:

- `sft_llamafactory/run_sft.ps1`
- `sft_llamafactory/run_compare_sft.ps1`
- `dpo_llamafactory/run_dpo.ps1`
- `dpo_llamafactory/run_compare_dpo.ps1`
- `dpo_llamafactory/run_score_dpo.ps1`
- `ppo_llamafactory/run_reward_model.ps1`
- `ppo_llamafactory/run_ppo.ps1`
- `ppo_llamafactory/run_compare_ppo.ps1`
- `ppo_llamafactory/run_score_reward_model.ps1`
- `ppo_trl/run_ppo.ps1`
- `grpo_trl/run_grpo.ps1`
- `grpo_trl/run_compare_grpo.ps1`
- `grpo_trl_numina/run_grpo.ps1`
- `grpo_verl/run_prepare_data.ps1`
- `grpo_verl/run_grpo.ps1`

The default Windows model path for the DPO/SFT/PPO demos is:

```powershell
E:\code\vibe\Qwen2.5-0.5B-Instruct
```

You can override it with:

```powershell
$env:MODEL_PATH="E:\code\vibe\Qwen2.5-0.5B-Instruct"
```

To compare the base model with the DPO LoRA adapter on prompts from `tiny_dpo.json`:

```powershell
cd E:\code\vibe\llm\rl\qwen_small_rlhf_demo\dpo_llamafactory
$env:MODEL_PATH="E:\code\vibe\Qwen2.5-0.5B-Instruct"
$env:ADAPTER_PATH="E:\code\vibe\llm\rl\qwen_small_rlhf_demo\dpo_llamafactory\outputs\dpo_lora"
.\run_compare_dpo.ps1
```

This writes a JSONL file containing `prompt`, `base_output`, `lora_output`, `chosen`, and `rejected`.

To score whether the DPO adapter assigns higher probability to `chosen` than `rejected`:

```powershell
cd E:\code\vibe\llm\rl\qwen_small_rlhf_demo\dpo_llamafactory
$env:MODEL_PATH="E:\code\vibe\Qwen2.5-0.5B-Instruct"
$env:ADAPTER_PATH="E:\code\vibe\llm\rl\qwen_small_rlhf_demo\dpo_llamafactory\outputs\dpo_lora"
.\run_score_dpo.ps1
```

This writes `outputs\dpo_score.json` with base-vs-LoRA margins for each sample.

## Recommended model choice

For the DPO/SFT/PPO demos, the recommended smallest instruct checkpoint is:

```bash
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir ./Qwen2.5-0.5B-Instruct
```

The scripts default to:

```bash
Qwen/Qwen2.5-0.5B-Instruct
```

If you actually downloaded the model somewhere else, set:

```bash
export MODEL_PATH=/your/model/path
```

If you later switch to a local path, set:

```bash
export MODEL_PATH=/your/model/path
```

The LLaMA-Factory demos default to the `qwen` template.

## Method Cheat Sheet

| Method | Training signal | Reward model needed | Best for |
| --- | --- | --- | --- |
| `SFT` | direct target answer | No | teaching a model to answer in a specific way |
| `DPO` | chosen vs rejected preference pairs | No | preference alignment without training a separate RM |
| `PPO` | online reward-driven policy updates | Often yes | classic RLHF pipelines with a learned reward model |
| `GRPO` | group-relative reward comparison | Often no | online RL with clear rule-based rewards |

In this repo, the practical takeaway is:

- `SFT` is the easiest way to force a visible behavior change
- `DPO` is a good middle ground for preference learning
- `PPO` is the longest pipeline because it usually needs reward modeling first
- `GRPO` is a nice fit when you can write a reward rule directly

## Smallest dataset choices

For low compute, the simplest route is to avoid large public datasets entirely and use local toy data:

- `SFT`: 8 instruction-output pairs
- `DPO`: 8 preference pairs
- `PPO (LLaMA-Factory)`: 8 prompts plus the same 8 preference pairs to train a tiny reward model first
- `PPO (TRL)`: 8 prompts with a rule-based reward
- `GRPO`: a small prompt set with a rule-based reward; in this repo it can also be aligned to the visible `淘宝 -> 无法回答。` behavior test

This is not enough for a meaningful aligned model, but it is ideal for verifying the training pipeline end to end.

## Suggested order

1. Run `sft_llamafactory` or `dpo_llamafactory` first
2. Run `grpo_trl` second
3. Try `grpo_verl` only if you want a more production-style RL stack and have a proper Linux/CUDA environment
4. Run `ppo_trl` or `ppo_llamafactory` last

`ppo_llamafactory` is the heaviest path because it needs a reward model checkpoint before PPO training starts.
`ppo_trl` is lighter because it uses a rule-based reward.

For the visible keyword-based behavior test, `sft_llamafactory` and `dpo_llamafactory` should show stronger changes than PPO.

The bundled SFT/DPO/PPO demo data includes a visible behavior test: several prompts containing `淘宝` prefer the response `无法回答。`, so you can quickly inspect whether the trained policy starts following that pattern.

To compare the base model with the SFT LoRA adapter on the bundled prompts:

```powershell
cd E:\code\vibe\llm\rl\qwen_small_rlhf_demo\sft_llamafactory
$env:MODEL_PATH="E:\code\vibe\Qwen2.5-0.5B-Instruct"
$env:ADAPTER_PATH="E:\code\vibe\llm\rl\qwen_small_rlhf_demo\sft_llamafactory\outputs\sft_lora"
.\run_compare_sft.ps1
```

To compare the base model with the PPO LoRA adapter on the bundled prompts:

```powershell
cd E:\code\vibe\llm\rl\qwen_small_rlhf_demo\ppo_llamafactory
$env:MODEL_PATH="E:\code\vibe\Qwen2.5-0.5B-Instruct"
$env:ADAPTER_PATH="E:\code\vibe\llm\rl\qwen_small_rlhf_demo\ppo_llamafactory\outputs\ppo_lora"
.\run_compare_ppo.ps1
```

This writes `outputs\ppo_compare.jsonl` with `prompt`, `base_output`, and `ppo_output`.

## References

- LLaMA-Factory supports Qwen, DPO, reward modeling, and PPO: [GitHub README](https://github.com/hiyouga/LlamaFactory)
- TRL quickstart includes DPO and GRPO examples with Qwen small models: [TRL Quickstart](https://huggingface.co/docs/trl/quickstart)
- TRL GRPO requires a dataset with a `prompt` column and supports custom reward functions: [GRPO Trainer docs](https://huggingface.co/docs/trl/grpo_trainer)

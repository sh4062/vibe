# Qwen Small RLHF Demo

This folder contains four minimal demos for a locally downloaded Qwen small model:

- `dpo_llamafactory/`: DPO with LLaMA-Factory
- `ppo_llamafactory/`: Reward modeling + PPO with LLaMA-Factory
- `ppo_trl/`: PPO with TRL and a rule-based reward
- `grpo_trl/`: GRPO with TRL

These demos are intentionally tiny so they are easier to run on limited hardware.

## Windows usage

PowerShell launchers are included:

- `dpo_llamafactory/run_dpo.ps1`
- `dpo_llamafactory/run_compare_dpo.ps1`
- `dpo_llamafactory/run_score_dpo.ps1`
- `ppo_llamafactory/run_reward_model.ps1`
- `ppo_llamafactory/run_ppo.ps1`
- `ppo_llamafactory/run_compare_ppo.ps1`
- `ppo_trl/run_ppo.ps1`
- `grpo_trl/run_grpo.ps1`

The default Windows model path is:

```powershell
E:\code\vibe\Qwen3.5-0.8B
```

You can override it with:

```powershell
$env:MODEL_PATH="E:\code\vibe\Qwen3.5-0.8B"
```

To compare the base model with the DPO LoRA adapter on prompts from `tiny_dpo.json`:

```powershell
cd E:\code\vibe\llm\rl\qwen_small_rlhf_demo\dpo_llamafactory
$env:MODEL_PATH="E:\code\vibe\Qwen3.5-0.8B"
$env:ADAPTER_PATH="E:\code\vibe\llm\rl\qwen_small_rlhf_demo\dpo_llamafactory\outputs\dpo_lora"
.\run_compare_dpo.ps1
```

This writes a JSONL file containing `prompt`, `base_output`, `lora_output`, `chosen`, and `rejected`.

To score whether the DPO adapter assigns higher probability to `chosen` than `rejected`:

```powershell
cd E:\code\vibe\llm\rl\qwen_small_rlhf_demo\dpo_llamafactory
$env:MODEL_PATH="E:\code\vibe\Qwen3.5-0.8B"
$env:ADAPTER_PATH="E:\code\vibe\llm\rl\qwen_small_rlhf_demo\dpo_llamafactory\outputs\dpo_lora"
.\run_score_dpo.ps1
```

This writes `outputs\dpo_score.json` with base-vs-LoRA margins for each sample.

## Recommended model choice

You said you already downloaded:

```bash
huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir ./Qwen3.5-0.8B
```

The scripts default to:

```bash
/Users/a1-6/Documents/code/vibe/Qwen3.5-0.8B
```

If you actually downloaded the model somewhere else, set:

```bash
export MODEL_PATH=/your/model/path
```

If you later switch to an instruct checkpoint, also set a chat template, for example:

```bash
export TEMPLATE=qwen3
```

For a base checkpoint, `default` is the safest starting point in these demos.

## Smallest dataset choices

For low compute, the simplest route is to avoid large public datasets entirely and use local toy data:

- `DPO`: 8 hand-written preference pairs
- `PPO (LLaMA-Factory)`: 8 prompts plus the same 8 preference pairs to train a tiny reward model first
- `PPO (TRL)`: 8 prompts with a rule-based reward
- `GRPO`: 12 arithmetic prompts with a rule-based exact-match reward

This is not enough for a meaningful aligned model, but it is ideal for verifying the training pipeline end to end.

## Suggested order

1. Run `dpo_llamafactory` first
2. Run `grpo_trl` second
3. Run `ppo_trl` or `ppo_llamafactory` last

`ppo_llamafactory` is the heaviest path because it needs a reward model checkpoint before PPO training starts.
`ppo_trl` is lighter because it uses a rule-based reward.

For `ppo_llamafactory`, a smaller instruct checkpoint such as `Qwen/Qwen2.5-0.5B-Instruct` is recommended over `Qwen3.5-0.8B` because PPO and reward-model tooling is currently more stable there.

The bundled PPO demo data also includes a visible behavior test: several prompts containing `淘宝` prefer the response `无法回答。`, so you can quickly inspect whether the trained policy starts following that pattern.

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

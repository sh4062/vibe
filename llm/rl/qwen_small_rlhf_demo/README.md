# Qwen Small RLHF Demo

This folder contains three minimal demos for a locally downloaded Qwen small model:

- `dpo_llamafactory/`: DPO with LLaMA-Factory
- `ppo_llamafactory/`: Reward modeling + PPO with LLaMA-Factory
- `grpo_trl/`: GRPO with TRL

These demos are intentionally tiny so they are easier to run on limited hardware.

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
- `PPO`: 8 prompts plus the same 8 preference pairs to train a tiny reward model first
- `GRPO`: 12 arithmetic prompts with a rule-based exact-match reward

This is not enough for a meaningful aligned model, but it is ideal for verifying the training pipeline end to end.

## Suggested order

1. Run `dpo_llamafactory` first
2. Run `grpo_trl` second
3. Run `ppo_llamafactory` last

`PPO` is the heaviest path because it needs a reward model checkpoint before PPO training starts.

## References

- LLaMA-Factory supports Qwen, DPO, reward modeling, and PPO: [GitHub README](https://github.com/hiyouga/LlamaFactory)
- TRL quickstart includes DPO and GRPO examples with Qwen small models: [TRL Quickstart](https://huggingface.co/docs/trl/quickstart)
- TRL GRPO requires a dataset with a `prompt` column and supports custom reward functions: [GRPO Trainer docs](https://huggingface.co/docs/trl/grpo_trainer)

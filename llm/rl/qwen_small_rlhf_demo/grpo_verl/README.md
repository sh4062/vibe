# GRPO with verl

This folder contains a small verl-based GRPO demo scaffold.

Default setup:

- model: `Qwen/Qwen2.5-0.5B-Instruct`
- algorithm: `GRPO`
- rollout backend: `vllm`
- tuning method: `LoRA`
- reward: exact-match reward on tiny arithmetic prompts

This path is more experimental than the SFT/DPO/PPO demos.

## Platform note

verl is much happier on Linux with CUDA than on Windows. If your main environment is Windows, it is usually better to run this demo on a Linux server or a Linux container with GPU access.

## What is included

- `data/tiny_math.jsonl`: 12 tiny arithmetic prompts
- `prepare_dataset.py`: converts the JSONL toy data into verl parquet files
- `reward_fn.py`: a custom reward that checks whether the generated answer matches the expected integer
- `run_prepare_data.ps1` / `run_prepare_data.sh`: build `train.parquet` and `val.parquet`
- `run_grpo.ps1` / `run_grpo.sh`: launch a conservative single-GPU GRPO run

## Suggested environment

Use a separate environment from the older `trl==0.9.6` / LLaMA-Factory PPO setup.

Typical packages you will need:

- `verl`
- `vllm`
- `torch`
- `transformers`
- `datasets`
- `pandas`
- `pyarrow`

## How to run

1. Prepare the parquet files:

```bash
cd llm/rl/qwen_small_rlhf_demo/grpo_verl
python prepare_dataset.py
```

or on PowerShell:

```powershell
cd E:\code\vibe\llm\rl\qwen_small_rlhf_demo\grpo_verl
.\run_prepare_data.ps1
```

2. Launch GRPO:

```bash
export MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
bash run_grpo.sh
```

or on PowerShell:

```powershell
$env:MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"
.\run_grpo.ps1
```

The launcher is intentionally conservative:

- `data.train_batch_size=8`
- `actor_rollout_ref.actor.ppo_mini_batch_size=8`
- `actor_rollout_ref.rollout.n=2`
- `actor_rollout_ref.rollout.gpu_memory_utilization=0.3`

That will still require a real CUDA environment, but it is much gentler than a typical verl recipe.

## Notes

- verl GRPO uses `python -m verl.trainer.main_ppo` with `algorithm.adv_estimator=grpo`
- the toy parquet data uses `prompt` and `ground_truth` columns
- the custom reward is configured through `custom_reward_function.path`
- this demo disables a learned reward model and relies only on the rule-based reward

## References

- [verl GitHub](https://github.com/verl-project/verl)
- [Qwen verl docs](https://qwen.readthedocs.io/en/v3.0/training/verl.html)
- [verl config explanation](https://verl.readthedocs.io/en/latest/examples/config.html)

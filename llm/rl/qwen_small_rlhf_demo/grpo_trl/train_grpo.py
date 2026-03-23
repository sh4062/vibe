import os
import re
import inspect

from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer


def _completion_to_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def refusal_reward(completions, ground_truth, **kwargs):
    rewards = []
    for completion, truth in zip(completions, ground_truth):
        text = _completion_to_text(completion).strip()
        if not text:
            rewards.append(0.0)
            continue

        reward = 0.0
        if truth in text:
            reward += 1.0

        # Encourage short, direct refusals for the Taobao prompts.
        if truth == "无法回答。":
            if len(text) <= 12:
                reward += 0.2
            if "淘宝" in text or "京东" in text:
                reward -= 0.5
        rewards.append(reward)
    return rewards


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.environ.get(
        "MODEL_PATH",
        "Qwen/Qwen2.5-0.5B-Instruct",
    )
    output_dir = os.environ.get(
        "OUTPUT_DIR",
        os.path.join(script_dir, "outputs", "grpo_lora"),
    )

    dataset = load_dataset(
        "json",
        data_files=os.path.join(script_dir, "data", "tiny_refusal.jsonl"),
        split="train",
    )

    config_kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 5e-6,
        "num_train_epochs": 3,
        "logging_steps": 1,
        "save_steps": 20,
        "report_to": "none",
        "bf16": False,
        "fp16": True,
    }
    optional_kwargs = {
        "max_prompt_length": 128,
        "max_completion_length": 32,
        "max_length": 160,
        "num_generations": 4,
        "use_vllm": False,
    }
    accepted = inspect.signature(GRPOConfig.__init__).parameters
    for key, value in optional_kwargs.items():
        if key in accepted:
            config_kwargs[key] = value

    config = GRPOConfig(**config_kwargs)

    trainer = GRPOTrainer(
        model=model_path,
        reward_funcs=refusal_reward,
        args=config,
        train_dataset=dataset,
        peft_config=LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    trainer.train()


if __name__ == "__main__":
    main()

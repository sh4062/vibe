import os
import re

from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer


def exact_match_reward(completions, ground_truth, **kwargs):
    rewards = []
    for completion, truth in zip(completions, ground_truth):
        text = completion.strip()
        match = re.search(r"-?\d+", text)
        pred = match.group(0) if match else ""
        rewards.append(1.0 if pred == truth else 0.0)
    return rewards


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.environ.get(
        "MODEL_PATH",
        "/Users/a1-6/Documents/code/vibe/Qwen3.5-0.8B",
    )
    output_dir = os.environ.get(
        "OUTPUT_DIR",
        os.path.join(script_dir, "outputs", "grpo_lora"),
    )

    dataset = load_dataset(
        "json",
        data_files=os.path.join(script_dir, "data", "tiny_math.jsonl"),
        split="train",
    )

    config = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        num_train_epochs=1,
        max_prompt_length=64,
        max_completion_length=16,
        logging_steps=1,
        save_steps=20,
        report_to="none",
        bf16=False,
        fp16=True,
    )

    trainer = GRPOTrainer(
        model=model_path,
        reward_funcs=exact_match_reward,
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

import json
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from trl import PPOConfig, PPOTrainer


class RuleRewardModel(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        scores = []
        for row in input_ids:
            text = self.tokenizer.decode(row, skip_special_tokens=True).lower()
            score = 0.0
            if "please" in text:
                score += 1.0
            if "sorry" in text:
                score += 1.0
            if "thank" in text:
                score += 0.5
            if len(text.split()) <= 40:
                score += 0.5
            if "cannot help" in text and "sorry" not in text:
                score -= 1.0
            scores.append(score)

        logits = torch.tensor(scores, device=input_ids.device, dtype=torch.float32).unsqueeze(-1)
        return SimpleNamespace(logits=logits)


def load_prompt_dataset(data_path: str) -> Dataset:
    with open(data_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return Dataset.from_list(rows)


def prepare_dataset(dataset: Dataset, tokenizer):
    def tokenize(row):
        encoded = tokenizer(
            row["prompt"],
            add_special_tokens=False,
            truncation=True,
            max_length=128,
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    return dataset.map(tokenize, remove_columns=dataset.column_names)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.environ.get("MODEL_PATH", r"E:\code\vibe\Qwen3.5-0.8B")
    data_path = os.environ.get("DATA_PATH", os.path.join(script_dir, "data", "tiny_prompts.json"))
    output_dir = os.environ.get("OUTPUT_DIR", os.path.join(script_dir, "outputs", "ppo_lora"))

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    torch_dtype = torch.float16 if use_cuda else torch.float32

    policy = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto" if use_cuda else None,
    )

    reward_model = RuleRewardModel(tokenizer)
    value_model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto" if use_cuda else None,
    )
    if use_cuda:
        reward_model = reward_model.cuda()

    dataset = prepare_dataset(load_prompt_dataset(data_path), tokenizer)

    config = PPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        num_ppo_epochs=1,
        total_episodes=len(dataset),
        num_mini_batches=1,
        response_length=48,
        logging_steps=1,
        save_steps=20,
        report_to="none",
        fp16=use_cuda,
        bf16=False,
    )

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    trainer = PPOTrainer(
        args=config,
        processing_class=tokenizer,
        model=policy,
        ref_model=None,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()


if __name__ == "__main__":
    main()

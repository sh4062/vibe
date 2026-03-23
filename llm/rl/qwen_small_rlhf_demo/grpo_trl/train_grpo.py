import os
import inspect
import json


def _plot_metric(log_history, key, output_path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    points = [(item.get("epoch"), item.get(key)) for item in log_history if key in item and item.get("epoch") is not None]
    points = [(x, y) for x, y in points if y is not None]
    if not points:
        return False

    xs, ys = zip(*points)
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o", linewidth=1.5)
    plt.xlabel("epoch")
    plt.ylabel(key)
    plt.title(key)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def _save_training_artifacts(trainer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    trainer.save_state()

    log_path = os.path.join(output_dir, "trainer_log.jsonl")
    with open(log_path, "w", encoding="utf-8") as f:
        for item in trainer.state.log_history:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    loss_path = os.path.join(output_dir, "training_loss.png")
    reward_path = os.path.join(output_dir, "training_reward.png")
    if _plot_metric(trainer.state.log_history, "loss", loss_path):
        print(f"Figure saved at: {loss_path}")
    if _plot_metric(trainer.state.log_history, "reward", reward_path):
        print(f"Figure saved at: {reward_path}")

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


def _extract_json(text: str):
    text = text.strip()
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def refusal_reward(completions, ground_truth, **kwargs):
    rewards = []
    for completion, truth in zip(completions, ground_truth):
        text = _completion_to_text(completion).strip()
        if not text:
            rewards.append(0.0)
            continue

        reward = 0.0
        parsed = _extract_json(text)
        truth_obj = json.loads(truth)
        truth_answer = truth_obj["answer"]

        # Strongly reward valid JSON with the expected schema.
        if parsed is not None:
            reward += 0.8
            if isinstance(parsed, dict) and "answer" in parsed:
                reward += 0.4
                answer = str(parsed["answer"]).strip()
                if answer == truth_answer:
                    reward += 1.0
                elif truth_answer in answer:
                    reward += 0.5
                if len(answer) <= 16:
                    reward += 0.1
                if truth_answer == "无法回答。" and ("淘宝" in answer or "京东" in answer):
                    reward -= 0.5
        else:
            # Small reward if the plain text still contains the target, but prefer JSON strongly.
            if truth_answer in text:
                reward += 0.2
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
        "learning_rate": 1e-5,
        "num_train_epochs": 3,
        "logging_steps": 1,
        "save_steps": 20,
        "report_to": "none",
        "bf16": False,
        "fp16": True,
    }
    optional_kwargs = {
        "max_prompt_length": 128,
        "max_completion_length": 48,
        "max_length": 176,
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
    _save_training_artifacts(trainer, output_dir)


if __name__ == "__main__":
    main()

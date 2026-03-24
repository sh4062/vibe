import inspect
import json
import os
import re
from typing import Any

from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think><answer> answer here </answer>."
)


def prepare_example(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
        "solution": example["solution"],
        "problem": example["problem"],
    }


def format_reward(completions, **kwargs):
    del kwargs
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    scores = []
    for completion in completions:
        content = completion[0]["content"] if isinstance(completion, list) else str(completion)
        scores.append(1.0 if re.match(pattern, content, flags=re.DOTALL) else 0.0)
    return scores


def _extract_answer_text(content: str) -> str:
    answer_match = re.search(r"<answer>(.*?)</answer>", content, flags=re.DOTALL)
    return answer_match.group(1).strip() if answer_match else content.strip()


def _fallback_math_match(pred_text: str, gold_text: str) -> float:
    pred = _extract_answer_text(pred_text)
    pred_numbers = re.findall(r"-?\d+(?:/\d+)?(?:\.\d+)?", pred)
    gold_numbers = re.findall(r"-?\d+(?:/\d+)?(?:\.\d+)?", gold_text)
    if not gold_numbers:
        return 1.0 if pred.strip() == gold_text.strip() else 0.0
    return 1.0 if pred_numbers and pred_numbers[-1] == gold_numbers[-1] else 0.0


try:
    from math_verify import LatexExtractionConfig, parse, verify

    def accuracy_reward(completions, solution, **kwargs):
        del kwargs
        rewards = []
        for completion, gold in zip(completions, solution):
            content = completion[0]["content"] if isinstance(completion, list) else str(completion)
            answer_text = _extract_answer_text(content)
            gold_parsed = parse(gold, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            pred_parsed = parse(answer_text, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) != 0:
                try:
                    rewards.append(float(verify(pred_parsed, gold_parsed)))
                except Exception:
                    rewards.append(0.0)
            else:
                rewards.append(_fallback_math_match(answer_text, gold))
        return rewards

except Exception:

    def accuracy_reward(completions, solution, **kwargs):
        del kwargs
        return [
            _fallback_math_match(completion[0]["content"] if isinstance(completion, list) else str(completion), gold)
            for completion, gold in zip(completions, solution)
        ]


def _plot_metric(log_history: list[dict[str, Any]], key: str, output_path: str) -> bool:
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


def save_training_artifacts(trainer, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    trainer.save_state()

    log_path = os.path.join(output_dir, "trainer_log.jsonl")
    with open(log_path, "w", encoding="utf-8") as f:
        for item in trainer.state.log_history:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    metric_to_name = {
        "loss": "training_loss.png",
        "reward": "training_reward.png",
        "eval_reward": "eval_reward.png",
        "rewards/format_reward/mean": "format_reward.png",
        "rewards/accuracy_reward/mean": "accuracy_reward.png",
        "eval_rewards/format_reward/mean": "eval_format_reward.png",
        "eval_rewards/accuracy_reward/mean": "eval_accuracy_reward.png",
    }
    for metric_key, file_name in metric_to_name.items():
        output_path = os.path.join(output_dir, file_name)
        if _plot_metric(trainer.state.log_history, metric_key, output_path):
            print(f"Figure saved at: {output_path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct")
    output_dir = os.environ.get("OUTPUT_DIR", os.path.join(script_dir, "outputs", "grpo_numina_lora"))
    dataset_id = os.environ.get("DATASET_ID", "AI-MO/NuminaMath-TIR")
    train_split = os.environ.get("TRAIN_SPLIT", "train[:1%]")
    eval_split = os.environ.get("EVAL_SPLIT", "test[:1%]")

    train_dataset = load_dataset(dataset_id, split=train_split)
    eval_dataset = load_dataset(dataset_id, split=eval_split)
    train_dataset = train_dataset.map(prepare_example, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(prepare_example, remove_columns=eval_dataset.column_names)

    config_kwargs = {
        "output_dir": output_dir,
        "learning_rate": 1e-5,
        "remove_unused_columns": False,
        "gradient_accumulation_steps": 16,
        "num_train_epochs": 1,
        "logging_steps": 10,
        "save_steps": 25,
        "report_to": "none",
    }
    optional_kwargs = {
        "bf16": True,
        "fp16": False,
        "max_completion_length": 160,
        "num_generations": 4,
        "per_device_train_batch_size": 1,
        "eval_strategy": "steps",
        "eval_steps": 25,
        "save_strategy": "steps",
        "max_prompt_length": 512,
        "max_length": 768,
    }

    accepted = inspect.signature(GRPOConfig.__init__).parameters
    for key, value in optional_kwargs.items():
        if key in accepted:
            config_kwargs[key] = value

    training_args = GRPOConfig(**config_kwargs)

    trainer = GRPOTrainer(
        model=model_path,
        reward_funcs=[format_reward, accuracy_reward],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    trainer.train()
    try:
        trainer.evaluate()
    except Exception as exc:
        print(f"Skipped explicit evaluation at the end: {exc}")
    save_training_artifacts(trainer, output_dir)


if __name__ == "__main__":
    main()

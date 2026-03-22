import gc
import json
import math
import os
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_samples(data_path: str) -> list[dict[str, Any]]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for idx, item in enumerate(data):
        prompt = item["conversations"][0]["value"]
        chosen = item["chosen"]["value"]
        rejected = item["rejected"]["value"]
        samples.append(
            {
                "id": idx,
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )
    return samples


def load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_path: str):
    model_kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = "auto"
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    if not torch.cuda.is_available():
        model.to("cpu")
    model.eval()
    return model


def cleanup_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_model_device(model) -> torch.device:
    return next(model.parameters()).device


@torch.inference_mode()
def continuation_logprob(model, tokenizer, prompt: str, continuation: str) -> dict[str, float]:
    device = get_model_device(model)

    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    full_ids = tokenizer(prompt + continuation, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)

    if full_ids.shape[1] <= prompt_ids.shape[1]:
        return {"sum_logprob": float("-inf"), "avg_logprob": float("-inf"), "token_count": 0}

    labels = full_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100

    outputs = model(input_ids=full_ids)
    logits = outputs.logits[:, :-1, :]
    target = full_ids[:, 1:]
    target_mask = labels[:, 1:] != -100

    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    selected = token_log_probs[target_mask]

    if selected.numel() == 0:
        return {"sum_logprob": float("-inf"), "avg_logprob": float("-inf"), "token_count": 0}

    sum_logprob = float(selected.sum().item())
    avg_logprob = float(selected.mean().item())
    return {
        "sum_logprob": sum_logprob,
        "avg_logprob": avg_logprob,
        "token_count": int(selected.numel()),
    }


def score_model(model, tokenizer, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for sample in samples:
        chosen_stats = continuation_logprob(model, tokenizer, sample["prompt"], sample["chosen"])
        rejected_stats = continuation_logprob(model, tokenizer, sample["prompt"], sample["rejected"])

        results.append(
            {
                "id": sample["id"],
                "prompt": sample["prompt"],
                "chosen": sample["chosen"],
                "rejected": sample["rejected"],
                "chosen_sum_logprob": chosen_stats["sum_logprob"],
                "rejected_sum_logprob": rejected_stats["sum_logprob"],
                "chosen_avg_logprob": chosen_stats["avg_logprob"],
                "rejected_avg_logprob": rejected_stats["avg_logprob"],
                "chosen_tokens": chosen_stats["token_count"],
                "rejected_tokens": rejected_stats["token_count"],
                "sum_margin": chosen_stats["sum_logprob"] - rejected_stats["sum_logprob"],
                "avg_margin": chosen_stats["avg_logprob"] - rejected_stats["avg_logprob"],
            }
        )
    return results


def summarize(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    avg_sum_margin = sum(row["sum_margin"] for row in rows) / len(rows)
    avg_avg_margin = sum(row["avg_margin"] for row in rows) / len(rows)
    preferred_count = sum(1 for row in rows if row["sum_margin"] > 0)
    return {
        "label": label,
        "samples": len(rows),
        "preferred_count": preferred_count,
        "preferred_ratio": preferred_count / len(rows),
        "mean_sum_margin": avg_sum_margin,
        "mean_avg_margin": avg_avg_margin,
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.environ.get("MODEL_PATH", r"E:\code\vibe\Qwen3.5-0.8B")
    adapter_path = os.environ.get(
        "ADAPTER_PATH",
        os.path.join(script_dir, "outputs", "dpo_lora"),
    )
    data_path = os.environ.get(
        "DATA_PATH",
        os.path.join(script_dir, "data", "tiny_dpo.json"),
    )
    output_path = os.environ.get(
        "SCORE_OUTPUT",
        os.path.join(script_dir, "outputs", "dpo_score.json"),
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    samples = load_samples(data_path)
    tokenizer = load_tokenizer(model_path)

    print(f"Loaded {len(samples)} samples from {data_path}")
    print("Scoring base model...")
    base_model = load_model(model_path)
    base_rows = score_model(base_model, tokenizer, samples)
    base_summary = summarize(base_rows, "base")
    cleanup_model(base_model)

    print("Scoring LoRA-adapted model...")
    adapted_base = load_model(model_path)
    adapted_model = PeftModel.from_pretrained(adapted_base, adapter_path)
    adapted_model.eval()
    lora_rows = score_model(adapted_model, tokenizer, samples)
    lora_summary = summarize(lora_rows, "lora")
    cleanup_model(adapted_model)

    merged_rows = []
    for base_row, lora_row in zip(base_rows, lora_rows):
        merged_rows.append(
            {
                "id": base_row["id"],
                "prompt": base_row["prompt"],
                "chosen": base_row["chosen"],
                "rejected": base_row["rejected"],
                "base_sum_margin": base_row["sum_margin"],
                "lora_sum_margin": lora_row["sum_margin"],
                "delta_sum_margin": lora_row["sum_margin"] - base_row["sum_margin"],
                "base_avg_margin": base_row["avg_margin"],
                "lora_avg_margin": lora_row["avg_margin"],
                "delta_avg_margin": lora_row["avg_margin"] - base_row["avg_margin"],
            }
        )

    output = {
        "base_summary": base_summary,
        "lora_summary": lora_summary,
        "rows": merged_rows,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(base_summary, ensure_ascii=False))
    print(json.dumps(lora_summary, ensure_ascii=False))
    print(f"Saved DPO score report to {output_path}")


if __name__ == "__main__":
    main()

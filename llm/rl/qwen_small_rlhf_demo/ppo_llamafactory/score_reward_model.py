import gc
import json
import os
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead


def load_samples(data_path: str) -> list[dict[str, Any]]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for idx, item in enumerate(data):
        rows.append(
            {
                "id": idx,
                "prompt": item["conversations"][0]["value"],
                "chosen": item["chosen"]["value"],
                "rejected": item["rejected"]["value"],
            }
        )
    return rows


def load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def model_kwargs():
    kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = "auto"
        kwargs["device_map"] = "auto"
    return kwargs


def load_base_model(model_path: str):
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs())
    if not torch.cuda.is_available():
        model.to("cpu")
    model.eval()
    return model


def _load_value_head_weights(model, adapter_path: str):
    for filename in ("value_head.bin", "value_head.safetensors"):
        value_head_path = os.path.join(adapter_path, filename)
        if not os.path.exists(value_head_path):
            continue
        if filename.endswith(".bin"):
            state_dict = torch.load(value_head_path, map_location="cpu")
        else:
            from safetensors.torch import load_file

            state_dict = load_file(value_head_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        return {
            "path": value_head_path,
            "missing": missing,
            "unexpected": unexpected,
        }
    return None


def load_reward_model(model_path: str, adapter_path: str):
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path, **model_kwargs())
    model.pretrained_model = PeftModel.from_pretrained(model.pretrained_model, adapter_path)
    vh_info = _load_value_head_weights(model, adapter_path)
    if not torch.cuda.is_available():
        model.to("cpu")
    model.eval()
    return model, vh_info


def cleanup_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_model_device(model) -> torch.device:
    return next(model.parameters()).device


def build_full_text(tokenizer, prompt: str, answer: str) -> str:
    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


@torch.inference_mode()
def lm_continuation_stats(model, tokenizer, prompt: str, continuation: str) -> dict[str, float]:
    device = get_model_device(model)
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = prompt_text + continuation

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    full_ids = tokenizer(full_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)

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

    return {
        "sum_logprob": float(selected.sum().item()),
        "avg_logprob": float(selected.mean().item()),
        "token_count": int(selected.numel()),
    }


@torch.inference_mode()
def reward_score(model, tokenizer, prompt: str, answer: str) -> float:
    device = get_model_device(model)
    full_text = build_full_text(tokenizer, prompt, answer)
    encoded = tokenizer(full_text, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    outputs = model(**encoded)

    if hasattr(outputs, "value") and outputs.value is not None:
        values = outputs.value
    elif isinstance(outputs, tuple) and len(outputs) >= 3:
        values = outputs[2]
    else:
        raise RuntimeError("Could not locate value head outputs from the reward model.")

    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        last_index = values.shape[1] - 1
    else:
        last_index = int(attention_mask[0].sum().item()) - 1
    return float(values[0, last_index].item())


def summarize(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    preferred_count = sum(1 for row in rows if row["score_margin"] > 0)
    return {
        "label": label,
        "samples": len(rows),
        "preferred_count": preferred_count,
        "preferred_ratio": preferred_count / len(rows),
        "mean_score_margin": sum(row["score_margin"] for row in rows) / len(rows),
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct")
    adapter_path = os.environ.get("ADAPTER_PATH", os.path.join(script_dir, "outputs", "reward_model_lora"))
    data_path = os.environ.get("DATA_PATH", os.path.join(script_dir, "data", "tiny_rm.json"))
    output_path = os.environ.get("SCORE_OUTPUT", os.path.join(script_dir, "outputs", "reward_model_score.json"))
    limit_samples = int(os.environ.get("LIMIT_SAMPLES", "0"))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    samples = load_samples(data_path)
    if limit_samples > 0:
        samples = samples[:limit_samples]
    tokenizer = load_tokenizer(model_path)

    print(f"Loaded {len(samples)} samples from {data_path}")

    print("Scoring base model LM margins...")
    base_model = load_base_model(model_path)
    base_rows = []
    for sample in samples:
        chosen = lm_continuation_stats(base_model, tokenizer, sample["prompt"], sample["chosen"])
        rejected = lm_continuation_stats(base_model, tokenizer, sample["prompt"], sample["rejected"])
        base_rows.append(
            {
                "id": sample["id"],
                "prompt": sample["prompt"],
                "chosen": sample["chosen"],
                "rejected": sample["rejected"],
                "score_margin": chosen["avg_logprob"] - rejected["avg_logprob"],
            }
        )
    base_summary = summarize(base_rows, "base_lm")
    cleanup_model(base_model)

    print("Scoring reward model chosen-vs-rejected preferences...")
    reward_model, value_head_info = load_reward_model(model_path, adapter_path)
    rm_rows = []
    for sample in samples:
        chosen_score = reward_score(reward_model, tokenizer, sample["prompt"], sample["chosen"])
        rejected_score = reward_score(reward_model, tokenizer, sample["prompt"], sample["rejected"])
        rm_rows.append(
            {
                "id": sample["id"],
                "prompt": sample["prompt"],
                "chosen": sample["chosen"],
                "rejected": sample["rejected"],
                "chosen_score": chosen_score,
                "rejected_score": rejected_score,
                "score_margin": chosen_score - rejected_score,
            }
        )
    rm_summary = summarize(rm_rows, "reward_model")
    cleanup_model(reward_model)

    output = {
        "value_head_info": value_head_info,
        "base_summary": base_summary,
        "reward_model_summary": rm_summary,
        "rows": rm_rows,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(base_summary, ensure_ascii=False))
    print(json.dumps(rm_summary, ensure_ascii=False))
    print(f"Saved reward model score report to {output_path}")


if __name__ == "__main__":
    main()

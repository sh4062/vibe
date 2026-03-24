import gc
import json
import os
import re

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "Reply strictly in this format: <think>reasoning</think><answer>final answer</answer>. "
    "Do not add any text before or after the tags."
)
FORMAT_PATTERN = re.compile(r"^<think>.*?</think>\s*<answer>.*?</answer>$", flags=re.DOTALL)


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


def cleanup_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_messages(problem: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]


@torch.inference_mode()
def generate_text(model, tokenizer, problem: str, max_new_tokens: int, do_sample: bool):
    device = next(model.parameters()).device
    messages = build_messages(problem)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=0.7 if do_sample else 1.0,
        top_p=0.9 if do_sample else 1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def format_match(text: str) -> bool:
    return bool(FORMAT_PATTERN.match(text.strip()))


def sample_many(model, tokenizer, problem: str, max_new_tokens: int, sample_count: int):
    outputs = []
    for _ in range(sample_count):
        text = generate_text(model, tokenizer, problem, max_new_tokens, do_sample=True)
        outputs.append(
            {
                "text": text,
                "format_match": format_match(text),
            }
        )
    return outputs


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct")
    adapter_path = os.environ.get("ADAPTER_PATH", os.path.join(script_dir, "outputs", "grpo_numina_lora"))
    dataset_id = os.environ.get("DATASET_ID", "AI-MO/NuminaMath-TIR")
    split = os.environ.get("INSPECT_SPLIT", "test[:5]")
    output_path = os.environ.get("INSPECT_OUTPUT", os.path.join(script_dir, "outputs", "inspect_generations.jsonl"))
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "256"))
    sample_count = int(os.environ.get("SAMPLE_COUNT", "4"))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dataset = load_dataset(dataset_id, split=split)
    tokenizer = load_tokenizer(model_path)

    print(f"Loaded {len(dataset)} samples from {dataset_id} ({split})")

    base_model = load_model(model_path)
    base_rows = []
    for item in dataset:
        base_greedy = generate_text(base_model, tokenizer, item["problem"], max_new_tokens, do_sample=False)
        base_samples = sample_many(base_model, tokenizer, item["problem"], max_new_tokens, sample_count)
        base_rows.append(
            {
                "problem": item["problem"],
                "solution": item["solution"],
                "base_greedy": base_greedy,
                "base_greedy_format_match": format_match(base_greedy),
                "base_samples": base_samples,
                "base_format_hits": sum(1 for sample in base_samples if sample["format_match"]),
                "base_sample_count": sample_count,
            }
        )
    cleanup_model(base_model)

    adapted_base = load_model(model_path)
    adapted_model = PeftModel.from_pretrained(adapted_base, adapter_path)
    adapted_model.eval()

    with open(output_path, "w", encoding="utf-8") as f:
        for row in base_rows:
            lo_ra_greedy = generate_text(adapted_model, tokenizer, row["problem"], max_new_tokens, do_sample=False)
            lo_ra_samples = sample_many(adapted_model, tokenizer, row["problem"], max_new_tokens, sample_count)
            record = {
                "problem": row["problem"],
                "solution": row["solution"],
                "system_prompt": SYSTEM_PROMPT,
                "base_greedy": row["base_greedy"],
                "base_greedy_format_match": row["base_greedy_format_match"],
                "base_samples": row["base_samples"],
                "base_format_hits": row["base_format_hits"],
                "base_sample_count": row["base_sample_count"],
                "grpo_greedy": lo_ra_greedy,
                "grpo_greedy_format_match": format_match(lo_ra_greedy),
                "grpo_samples": lo_ra_samples,
                "grpo_format_hits": sum(1 for sample in lo_ra_samples if sample["format_match"]),
                "grpo_sample_count": sample_count,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    cleanup_model(adapted_model)
    print(f"Saved generation inspection results to {output_path}")


if __name__ == "__main__":
    main()

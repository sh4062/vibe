import gc
import json
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


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_tokenizer(model_path: str):
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def load_model(model_path: str):
    model_kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = "auto"
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    if not torch.cuda.is_available():
        model.to(get_device())
    model.eval()
    return model


def build_inputs(tokenizer, prompt: str, device: str):
    chat_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(chat_text, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


@torch.inference_mode()
def generate_text(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    device = next(model.parameters()).device
    inputs = build_inputs(tokenizer, prompt, device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def cleanup_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.environ.get("MODEL_PATH", r"E:\code\vibe\Qwen2.5-0.5B-Instruct")
    adapter_path = os.environ.get(
        "ADAPTER_PATH",
        os.path.join(script_dir, "outputs", "dpo_lora"),
    )
    data_path = os.environ.get(
        "DATA_PATH",
        os.path.join(script_dir, "data", "tiny_dpo.json"),
    )
    output_path = os.environ.get(
        "COMPARE_OUTPUT",
        os.path.join(script_dir, "outputs", "dpo_compare.jsonl"),
    )
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "64"))
    limit_samples = int(os.environ.get("LIMIT_SAMPLES", "0"))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    samples = load_samples(data_path)
    if limit_samples > 0:
        samples = samples[:limit_samples]
    tokenizer = load_tokenizer(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loaded {len(samples)} samples from {data_path}")
    print(f"Base model: {model_path}")
    print(f"Adapter: {adapter_path}")

    print("Generating base model outputs...")
    base_model = load_model(model_path)
    base_outputs = [
        generate_text(base_model, tokenizer, sample["prompt"], max_new_tokens)
        for sample in samples
    ]
    cleanup_model(base_model)

    print("Generating LoRA-adapted outputs...")
    adapted_base = load_model(model_path)
    adapted_model = PeftModel.from_pretrained(adapted_base, adapter_path)
    adapted_model.eval()
    adapted_outputs = [
        generate_text(adapted_model, tokenizer, sample["prompt"], max_new_tokens)
        for sample in samples
    ]
    cleanup_model(adapted_model)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample, base_output, adapted_output in zip(samples, base_outputs, adapted_outputs):
            record = {
                "id": sample["id"],
                "prompt": sample["prompt"],
                "base_output": base_output,
                "lora_output": adapted_output,
                "chosen": sample["chosen"],
                "rejected": sample["rejected"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved comparison results to {output_path}")


if __name__ == "__main__":
    main()

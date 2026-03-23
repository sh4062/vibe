import gc
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_samples(data_path: str):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        {
            "prompt": item["messages"][0]["content"],
            "target": item["messages"][1]["content"],
        }
        for item in data
    ]


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


@torch.inference_mode()
def generate_text(model, tokenizer, prompt: str, max_new_tokens: int):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.environ.get("MODEL_PATH", r"E:\code\vibe\Qwen2.5-0.5B-Instruct")
    adapter_path = os.environ.get("ADAPTER_PATH", os.path.join(script_dir, "outputs", "sft_lora"))
    data_path = os.environ.get("DATA_PATH", os.path.join(script_dir, "data", "tiny_sft.json"))
    output_path = os.environ.get("COMPARE_OUTPUT", os.path.join(script_dir, "outputs", "sft_compare.jsonl"))
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "48"))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    samples = load_samples(data_path)
    tokenizer = load_tokenizer(model_path)

    base_model = load_model(model_path)
    base_outputs = [generate_text(base_model, tokenizer, item["prompt"], max_new_tokens) for item in samples]
    cleanup_model(base_model)

    adapted_base = load_model(model_path)
    adapted_model = PeftModel.from_pretrained(adapted_base, adapter_path)
    adapted_model.eval()
    adapted_outputs = [generate_text(adapted_model, tokenizer, item["prompt"], max_new_tokens) for item in samples]
    cleanup_model(adapted_model)

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, (item, base_output, sft_output) in enumerate(zip(samples, base_outputs, adapted_outputs)):
            f.write(
                json.dumps(
                    {
                        "id": idx,
                        "prompt": item["prompt"],
                        "target": item["target"],
                        "base_output": base_output,
                        "sft_output": sft_output,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()

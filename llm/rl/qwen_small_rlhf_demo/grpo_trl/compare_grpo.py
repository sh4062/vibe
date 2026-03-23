import gc
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_samples(data_path: str):
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            samples.append(
                {
                    "id": idx,
                    "prompt": item["prompt"],
                    "target": item["ground_truth"],
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


def cleanup_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_inputs(tokenizer, prompt: str, device: torch.device):
    chat_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(chat_text, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


@torch.inference_mode()
def generate_text(model, tokenizer, prompt: str, max_new_tokens: int, do_sample: bool, temperature: float, top_p: float):
    device = next(model.parameters()).device
    inputs = build_inputs(tokenizer, prompt, device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def sample_many(model, tokenizer, prompt: str, max_new_tokens: int, sample_count: int):
    return [
        generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        for _ in range(sample_count)
    ]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct")
    adapter_path = os.environ.get("ADAPTER_PATH", os.path.join(script_dir, "outputs", "grpo_lora"))
    data_path = os.environ.get("DATA_PATH", os.path.join(script_dir, "data", "tiny_refusal.jsonl"))
    output_path = os.environ.get("COMPARE_OUTPUT", os.path.join(script_dir, "outputs", "grpo_compare.jsonl"))
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "48"))
    sample_count = int(os.environ.get("SAMPLE_COUNT", "3"))
    limit_samples = int(os.environ.get("LIMIT_SAMPLES", "0"))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    samples = load_samples(data_path)
    if limit_samples > 0:
        samples = samples[:limit_samples]
    tokenizer = load_tokenizer(model_path)

    print(f"Loaded {len(samples)} samples from {data_path}")
    print(f"Base model: {model_path}")
    print(f"Adapter: {adapter_path}")

    print("Generating base model outputs...")
    base_model = load_model(model_path)
    base_greedy_outputs = [
        generate_text(
            base_model,
            tokenizer,
            sample["prompt"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
        )
        for sample in samples
    ]
    base_sample_outputs = [
        sample_many(base_model, tokenizer, sample["prompt"], max_new_tokens, sample_count)
        for sample in samples
    ]
    cleanup_model(base_model)

    print("Generating GRPO-adapted outputs...")
    adapted_base = load_model(model_path)
    adapted_model = PeftModel.from_pretrained(adapted_base, adapter_path)
    adapted_model.eval()
    adapted_greedy_outputs = [
        generate_text(
            adapted_model,
            tokenizer,
            sample["prompt"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
        )
        for sample in samples
    ]
    adapted_sample_outputs = [
        sample_many(adapted_model, tokenizer, sample["prompt"], max_new_tokens, sample_count)
        for sample in samples
    ]
    cleanup_model(adapted_model)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample, base_greedy, base_samples, adapted_greedy, adapted_samples in zip(
            samples,
            base_greedy_outputs,
            base_sample_outputs,
            adapted_greedy_outputs,
            adapted_sample_outputs,
        ):
            record = {
                "id": sample["id"],
                "prompt": sample["prompt"],
                "target": sample["target"],
                "base_greedy": base_greedy,
                "grpo_greedy": adapted_greedy,
                "base_samples": base_samples,
                "grpo_samples": adapted_samples,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved comparison results to {output_path}")


if __name__ == "__main__":
    main()

"""Merge a Prime Intellect hosted training LoRA adapter into a base model.

NOTE: Do NOT use peft.PeftModel.from_pretrained() for adapters downloaded
from Prime's hosted training. Their adapter key format (model.layers.X...)
doesn't match peft's expected format (base_model.model.model.layers.X...),
resulting in a silent no-op merge. This script applies the LoRA weights
manually instead.

Usage:
    python merge_lora.py \\
        --base-model Qwen/Qwen3-4B-Instruct-2507 \\
        --adapter-path /path/to/adapter/ \\
        --output-path /path/to/merged-model/
"""

import argparse
import json

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge(base_model: str, adapter_path: str, output_path: str) -> None:
    with open(f"{adapter_path}/adapter_config.json") as f:
        cfg = json.load(f)
    scaling = cfg["lora_alpha"] / cfg["r"]
    print(f"LoRA config: r={cfg['r']}, alpha={cfg['lora_alpha']}, scaling={scaling}")

    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16)
    state_dict = model.state_dict()

    print(f"Loading adapter: {adapter_path}")
    adapter = load_file(f"{adapter_path}/adapter_model.safetensors")

    applied = 0
    for key, tensor in adapter.items():
        if "lora_A" not in key:
            continue
        base_key = key.replace(".lora_A.weight", ".weight")
        b_key = key.replace("lora_A", "lora_B")

        if base_key not in state_dict:
            print(f"  SKIP: {base_key} not in base model")
            continue
        if b_key not in adapter:
            print(f"  SKIP: {b_key} not in adapter")
            continue

        A = tensor.float()
        B = adapter[b_key].float()
        state_dict[base_key] = (state_dict[base_key].float() + (B @ A) * scaling).to(torch.bfloat16)
        applied += 1

    print(f"Applied LoRA to {applied} weight matrices")

    model.load_state_dict(state_dict)
    print(f"Saving merged model to {output_path}")
    model.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(output_path)
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    merge(args.base_model, args.adapter_path, args.output_path)

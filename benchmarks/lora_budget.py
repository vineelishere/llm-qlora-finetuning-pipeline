"""
LoRA parameter budget.

Before spending GPU hours, know exactly how much of the model you are training.
This reads only the model *architecture* (no weights are downloaded), instantiates
it on PyTorch's meta device, and counts the LoRA matrices that `lora.r` would add
to every configured `target_modules` entry.

Output: total params, trainable params, trainable ratio, and the on-disk adapter
size you can expect to shard, upload and load.

Usage:
    python benchmarks/lora_budget.py --config configs/llm_finetune.yaml
    python benchmarks/lora_budget.py --config configs/vlm_finetune.yaml --model Qwen/Qwen2.5-0.5B-Instruct
"""

import sys
import os
import json
import argparse
import datetime

import torch
from transformers import AutoConfig, AutoModelForCausalLM

try:
    from transformers import AutoModelForVision2Seq
except ImportError:  # transformers >= 5 renamed the vision-to-text auto class
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "utils"))
from config_loader import load_config  # noqa: E402


def lora_params_for_module(rank: int, in_features: int, out_features: int) -> int:
    """
    Parameters added by one LoRA-wrapped linear layer: A (r x in) + B (out x r).

    With `bias: none` no adapter biases are trained, so this is exact.
    """
    return rank * in_features + rank * out_features


def count_budget(model_name: str, lora_cfg: dict) -> dict:
    """Instantiate the architecture on the meta device and count LoRA vs total parameters."""
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    with torch.device("meta"):  # no weights allocated: shape metadata only
        try:
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        except ValueError:
            # Vision-language configs load through the image-to-text auto class instead
            model = AutoModelForVision2Seq.from_config(config, trust_remote_code=True)

    rank = lora_cfg.get("r", 16)
    targets = set(lora_cfg.get("target_modules") or [])
    if not targets:
        raise ValueError("lora.target_modules must list at least one module name")

    total_params = sum(p.numel() for p in model.parameters())

    trainable = 0
    per_module = {}
    matched = 0

    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in targets or not hasattr(module, "in_features"):
            continue
        matched += 1
        added = lora_params_for_module(rank, module.in_features, module.out_features)
        trainable += added
        per_module[leaf] = per_module.get(leaf, 0) + added

    if matched == 0:
        raise ValueError(
            f"No layers matched target_modules {sorted(targets)} in {model_name}; "
            "check the naming for this architecture."
        )

    ratio = trainable / total_params * 100

    return {
        "model": model_name,
        "rank": rank,
        "lora_alpha": lora_cfg.get("lora_alpha"),
        "wrapped_layers": matched,
        "total_params": total_params,
        "trainable_params": trainable,
        "trainable_percent": round(ratio, 3),
        "adapter_size_mb_bf16": round(trainable * 2 / (1024 * 1024), 2),
        "per_module_trainable_params": per_module,
    }


def main():
    parser = argparse.ArgumentParser(description="Report the LoRA trainable-parameter budget")
    parser.add_argument("--config", type=str, default="configs/llm_finetune.yaml")
    parser.add_argument("--model", type=str, default=None, help="Override the config's base_model")
    parser.add_argument("--output-dir", type=str, default="benchmarks/results")
    args = parser.parse_args()

    config = load_config(args.config)
    model_name = args.model or config["model"]["base_model"]

    budget = count_budget(model_name, config["lora"])
    budget["config_file"] = args.config
    budget["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")

    print(f"\n{model_name} · LoRA r={budget['rank']}, alpha={budget['lora_alpha']}")
    print(f"  wrapped layers      : {budget['wrapped_layers']}")
    print(f"  total parameters    : {budget['total_params']:,}")
    print(f"  trainable parameters: {budget['trainable_params']:,} ({budget['trainable_percent']}%)")
    print(f"  adapter on disk     : ~{budget['adapter_size_mb_bf16']} MB (bf16)")

    os.makedirs(args.output_dir, exist_ok=True)
    slug = model_name.split("/")[-1].lower()
    out_path = os.path.join(args.output_dir, f"lora_budget_{slug}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(budget, f, indent=2)

    print(f"  saved               : {out_path}")


if __name__ == "__main__":
    main()

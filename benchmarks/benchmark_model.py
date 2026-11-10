"""
Single-model serving benchmark.

Measures what actually decides whether a checkpoint is deployable: how long it
takes to load, how much memory it occupies, how fast it generates, and how well
it predicts held-out text. Runs on CPU or GPU, so the same script produces a
baseline before any quantization work starts.

Usage:
    python benchmarks/benchmark_model.py --model Qwen/Qwen2.5-0.5B-Instruct
    python benchmarks/benchmark_model.py --model outputs/merged_model --runs 5 --max-new-tokens 128

The JSON artifact it writes is the source of the numbers quoted in README.md -
measurements are never hand-entered.
"""

import sys
import os
import json
import time
import argparse
import platform
import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "evaluation"))
from metrics import (  # noqa: E402
    environment_fingerprint,
    folder_size_mb,
    peak_rss_mb,
    summarize_latencies,
    tokens_per_second,
)

# Fixed prompts keep runs comparable across machines and checkpoints
DEFAULT_PROMPTS = [
    "Explain the difference between LoRA and QLoRA in two sentences.",
    "Write a Python function that reverses a linked list.",
    "Summarize why quantizing a language model reduces serving cost.",
]

# Held-out instruction/response pairs for perplexity (Dolly-15k style template)
DEFAULT_EVAL_TEXTS = [
    "### Instruction:\nWhat is the capital of France?\n\n### Response:\nParis is the capital of France.",
    "### Instruction:\nExplain what a transformer is.\n\n### Response:\nA transformer is a neural network architecture that uses self-attention to weigh the relevance of every token against every other token in a sequence.",
    "### Instruction:\nName three benefits of containerizing a model server.\n\n### Response:\nReproducible environments, simpler dependency isolation, and the same artifact running identically on a laptop and in production.",
    "### Instruction:\nWhat does the learning rate control during fine-tuning?\n\n### Response:\nThe size of each weight update; too high diverges, too low trains slowly.",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark a causal LM's serving characteristics")
    parser.add_argument("--model", type=str, required=True, help="HF model id or local checkpoint directory")
    parser.add_argument("--runs", type=int, default=3, help="Timed generations per prompt (after 1 warmup)")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--threads", type=int, default=None, help="Force torch CPU thread count")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Weight dtype. 'auto' = bf16 on GPU, fp32 on CPU",
    )
    parser.add_argument("--skip-perplexity", action="store_true")
    parser.add_argument("--output-dir", type=str, default="benchmarks/results")
    parser.add_argument("--label", type=str, default=None, help="Name for this run in the artifact")
    return parser.parse_args()


def resolve_dtype(dtype_arg: str, device: str):
    """bf16/fp16 on GPU, fp32 on CPU - half precision matmuls are slow on most CPUs."""
    if dtype_arg != "auto":
        return getattr(torch, dtype_arg)
    return torch.bfloat16 if device == "cuda" else torch.float32


def load_model(model_name: str, threads: int = None, dtype_arg: str = "auto"):
    """Load tokenizer + model, returning them with load time and the device/dtype used."""
    if threads:
        torch.set_num_threads(threads)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = resolve_dtype(dtype_arg, device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
    load_seconds = time.perf_counter() - start

    model.eval()
    return model, tokenizer, load_seconds, device, str(dtype).replace("torch.", "")


def build_prompt(tokenizer, instruction: str) -> str:
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


@torch.no_grad()
def timed_generation(model, tokenizer, prompt: str, max_new_tokens: int, device: str):
    """Run one generation and return (seconds, new_token_count, text)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_tokens = inputs["input_ids"].shape[1]

    start = time.perf_counter()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy decoding: same work every run, so timings are comparable
        pad_token_id=tokenizer.pad_token_id,
    )
    seconds = time.perf_counter() - start

    new_tokens = output_ids.shape[1] - prompt_tokens
    text = tokenizer.decode(output_ids[0][prompt_tokens:], skip_special_tokens=True)

    return seconds, new_tokens, text.strip()


@torch.no_grad()
def calculate_perplexity(model, tokenizer, texts, device: str) -> float:
    """Token-level perplexity over fixed held-out texts (lower = better fit)."""
    total_loss, counted = 0.0, 0

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        loss = model(**inputs, labels=inputs["input_ids"]).loss
        if not torch.isnan(loss):
            total_loss += loss.item()
            counted += 1

    if counted == 0:
        return float("nan")

    return round(float(torch.exp(torch.tensor(total_loss / counted))), 4)


def benchmark(args) -> dict:
    model, tokenizer, load_seconds, device, dtype = load_model(args.model, args.threads, args.dtype)

    print(f"Loaded {args.model} on {device} ({dtype}) in {load_seconds:.2f}s")
    print(f"Resident memory after load: {peak_rss_mb()} MB\n")

    # One warmup pass so lazy kernel/allocator setup isn't billed to the first timed run
    timed_generation(
        model, tokenizer, build_prompt(tokenizer, DEFAULT_PROMPTS[0]), args.max_new_tokens, device
    )

    latencies, total_tokens = [], 0
    per_prompt = {}

    for instruction in DEFAULT_PROMPTS:
        prompt = build_prompt(tokenizer, instruction)
        prompt_latencies, prompt_tokens = [], 0

        for _ in range(args.runs):
            seconds, new_tokens, _text = timed_generation(
                model, tokenizer, prompt, args.max_new_tokens, device
            )
            prompt_latencies.append(seconds * 1000)
            prompt_tokens += new_tokens

        total_seconds = sum(prompt_latencies) / 1000
        prompt_tokens = max(prompt_tokens, 1)
        per_prompt[instruction] = {
            "latency": summarize_latencies(prompt_latencies),
            "tokens_per_second": tokens_per_second(prompt_tokens, total_seconds),
        }
        latencies.extend(prompt_latencies)
        total_tokens += prompt_tokens
        print(
            f"  {prompt_tokens} tokens over {args.runs} runs -> "
            f"{per_prompt[instruction]['tokens_per_second']:.2f} tok/s "
            f"(p50 {per_prompt[instruction]['latency']['p50_ms']:.0f} ms)"
        )

    total_decode_seconds = sum(latencies) / 1000
    perplexity = None
    if not args.skip_perplexity:
        perplexity = calculate_perplexity(model, tokenizer, DEFAULT_EVAL_TEXTS, device)
        print(f"  perplexity on {len(DEFAULT_EVAL_TEXTS)} held-out texts: {perplexity}")

    result = {
        "label": args.label or args.model.split("/")[-1],
        "model": args.model,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "environment": environment_fingerprint(device=device, dtype=dtype),
        "config": {
            "runs_per_prompt": args.runs,
            "max_new_tokens": args.max_new_tokens,
            "decoding": "greedy",
            "prompts": len(DEFAULT_PROMPTS),
        },
        "load_time_seconds": round(load_seconds, 2),
        "artifact_size_mb": folder_size_mb(args.model) if os.path.isdir(args.model) else None,
        "peak_rss_mb": peak_rss_mb(),
        "aggregate": {
            "generated_tokens": total_tokens,
            "decode_seconds": round(total_decode_seconds, 3),
            "tokens_per_second": tokens_per_second(total_tokens, total_decode_seconds),
            "latency": summarize_latencies(latencies),
        },
        "per_prompt": per_prompt,
        "perplexity": perplexity,
    }

    return result


def print_markdown(result: dict) -> None:
    env = result["environment"]
    agg = result["aggregate"]

    print("\n| Metric | Value |")
    print("|---|---|")
    print(f"| Model | `{result['model']}` |")
    print(f"| Device / dtype | {env['device']} / {env['dtype']} |")
    print(f"| Load time | {result['load_time_seconds']} s |")
    print(f"| Peak resident memory | {result['peak_rss_mb']} MB |")
    print(f"| Decode throughput | {agg['tokens_per_second']} tok/s |")
    print(f"| Latency p50 / p95 | {agg['latency']['p50_ms']} ms / {agg['latency']['p95_ms']} ms |")
    print(f"| Perplexity | {result['perplexity']} |")


def main():
    args = parse_args()
    result = benchmark(args)

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = args.label or args.model.split("/")[-1]
    out_path = os.path.join(args.output_dir, f"{slug}_{stamp}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print_markdown(result)
    print(f"\nSaved artifact: {out_path}")
    print(f"Machine: {platform.processor() or 'unknown CPU'} | {result['environment']['cpu_count']} logical cores")


if __name__ == "__main__":
    main()

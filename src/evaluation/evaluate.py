"""
Evaluation report for the fine-tuning + quantization pipeline.

Produces one JSON artifact (`outputs/quantization_report.json` by default) with:

  - perplexity: base model vs. fine-tuned model on held-out instruction samples
  - serving: load time, peak memory, latency p50/p95 and decode throughput
  - quantization: on-disk size of every artifact plus reduction vs. the FP16 merge
  - environment: device, dtype, torch/transformers versions, thread count

The environment block matters: throughput and memory are meaningless without it,
and it is what makes two runs comparable months apart.

Usage:
    python src/evaluation/evaluate.py --config configs/quantization.yaml
"""

import sys
import os
import json
import time
import argparse
import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__)))
from config_loader import load_config
from logger import get_logger
from metrics import (
    compression_ratio_mb,
    environment_fingerprint,
    folder_size_mb,
    peak_rss_mb,
    summarize_latencies,
    tokens_per_second,
)

logger = get_logger(__name__)

MERGED_DIR = "outputs/merged_model"
SPEED_PROMPT = "### Instruction:\nExplain why quantization reduces serving cost.\n\n### Response:\n"


def parse_args():
    parser = argparse.ArgumentParser(description="Model evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/quantization.yaml",
        help="Path to quantization config YAML file",
    )
    parser.add_argument(
        "--num_eval_samples",
        type=int,
        default=50,
        help="Number of samples to use for the perplexity calculation",
    )
    parser.add_argument(
        "--benchmark_runs",
        type=int,
        default=5,
        help="Timed generations used for the latency/throughput measurement",
    )
    return parser.parse_args()


def get_device_and_dtype():
    """bf16 on GPU, fp32 on CPU - bf16 matmuls are pathologically slow on most CPUs."""
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


def load_model_for_eval(model_path: str, device: str, dtype):
    """Load a model, returning it with its load time in seconds."""
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(device)
    load_seconds = time.perf_counter() - start

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer, load_seconds


def calculate_perplexity(model, tokenizer, eval_texts: list, device: str) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for text in eval_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            if not torch.isnan(loss):
                total_loss += loss.item()
                total_samples += 1

    avg_loss = total_loss / max(total_samples, 1)
    return round(float(torch.exp(torch.tensor(avg_loss))), 4)


def measure_serving(model, tokenizer, device: str, num_runs: int = 5, max_new_tokens: int = 50) -> dict:
    """
    Measure decode latency and throughput on a fixed prompt.

    One untimed warmup absorbs lazy setup; p50/p95 are reported because a single
    cold request distorts the mean.
    """
    model.eval()
    inputs = tokenizer(SPEED_PROMPT, return_tensors="pt").to(device)
    prompt_length = inputs["input_ids"].shape[1]

    def run_once():
        with torch.no_grad():
            start = time.perf_counter()
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            return time.perf_counter() - start, output.shape[1] - prompt_length

    run_once()  # warmup, discarded

    latencies_ms = []
    total_tokens = 0
    total_seconds = 0.0

    for _ in range(num_runs):
        seconds, new_tokens = run_once()
        latencies_ms.append(seconds * 1000)
        total_tokens += max(new_tokens, 1)
        total_seconds += seconds

    return {
        "latency": summarize_latencies(latencies_ms),
        "tokens_per_second": tokens_per_second(total_tokens, total_seconds),
        "max_new_tokens": max_new_tokens,
    }


def compare_base_vs_finetuned(
    base_model_name: str,
    finetuned_model_dir: str,
    eval_dataset_name: str,
    num_samples: int,
    benchmark_runs: int,
) -> dict:
    logger.info("Loading evaluation samples...")
    eval_data = load_dataset(eval_dataset_name, split="train").select(range(num_samples))
    eval_texts = [
        f"### Instruction:\n{ex.get('instruction', '')}\n\n### Response:\n{ex.get('response', '')}"
        for ex in eval_data
    ]

    device, dtype = get_device_and_dtype()
    logger.info(f"Evaluating on {device} ({dtype})")

    # ---------- Base model ----------
    logger.info(f"Loading base model: {base_model_name}")
    base_model, base_tokenizer, base_load_seconds = load_model_for_eval(base_model_name, device, dtype)

    logger.info("Calculating base model perplexity...")
    base_perplexity = calculate_perplexity(base_model, base_tokenizer, eval_texts, device)
    logger.info(f"Base model perplexity: {base_perplexity:.4f} (loaded in {base_load_seconds:.1f}s)")

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------- Fine-tuned model ----------
    logger.info(f"Loading fine-tuned model: {finetuned_model_dir}")
    finetuned_model, finetuned_tokenizer, finetuned_load_seconds = load_model_for_eval(
        finetuned_model_dir, device, dtype
    )

    logger.info("Calculating fine-tuned model perplexity...")
    finetuned_perplexity = calculate_perplexity(finetuned_model, finetuned_tokenizer, eval_texts, device)
    logger.info(f"Fine-tuned model perplexity: {finetuned_perplexity:.4f}")

    logger.info(f"Measuring serving performance ({benchmark_runs} timed runs)...")
    serving = measure_serving(finetuned_model, finetuned_tokenizer, device, num_runs=benchmark_runs)
    serving["load_time_seconds"] = round(finetuned_load_seconds, 2)
    serving["peak_rss_mb"] = peak_rss_mb()

    del finetuned_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    improvement_pct = ((base_perplexity - finetuned_perplexity) / base_perplexity) * 100

    return {
        "base_model_perplexity": base_perplexity,
        "finetuned_model_perplexity": finetuned_perplexity,
        "improvement_percent": round(improvement_pct, 2),
        "base_load_time_seconds": round(base_load_seconds, 2),
        "eval_samples": len(eval_texts),
        "serving": serving,
    }


def compare_quantization_methods(config: dict) -> dict:
    """Size of each artifact on disk, with reduction measured against the FP16 merge."""
    results = {}

    if os.path.isdir(MERGED_DIR):
        reference_mb = folder_size_mb(MERGED_DIR)
        results["original_merged_fp16"] = {"size_mb": reference_mb}
    else:
        reference_mb = None

    bnb_dir = config["bitsandbytes"]["output_dir"]
    if os.path.isdir(bnb_dir):
        # On-the-fly quantization: tokenizer + config only, no quantized weights saved
        results["bitsandbytes_4bit"] = {
            "size_mb": folder_size_mb(bnb_dir),
            "note": "quantized at load time; size is metadata only",
        }

    gguf_dir = config["gguf"]["output_dir"]
    if os.path.isdir(gguf_dir):
        for level in config["gguf"]["quant_levels"]:
            gguf_file = os.path.join(gguf_dir, f"model-{level}.gguf")
            if os.path.isfile(gguf_file):
                size_mb = round(os.path.getsize(gguf_file) / (1024 * 1024), 2)
                entry = {"size_mb": size_mb}
                if reference_mb:
                    entry["reduction_vs_fp16_percent"] = compression_ratio_mb(reference_mb, size_mb)
                results[f"gguf_{level}"] = entry

    awq_dir = config["awq"]["output_dir"]
    if os.path.isdir(awq_dir):
        size_mb = folder_size_mb(awq_dir)
        entry = {"size_mb": size_mb}
        if reference_mb:
            entry["reduction_vs_fp16_percent"] = compression_ratio_mb(reference_mb, size_mb)
        results["awq_4bit"] = entry

    return results


def main():
    args = parse_args()
    config = load_config(args.config)
    device, dtype = get_device_and_dtype()

    report = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "base_model": config["input"]["base_model"],
        "config_file": args.config,
        "environment": environment_fingerprint(device=device, dtype=str(dtype).replace("torch.", "")),
    }

    # ---------- Part 1: Perplexity + serving benchmark ----------
    logger.info("=" * 60)
    logger.info("Running perplexity comparison (base vs fine-tuned)...")
    logger.info("=" * 60)

    try:
        report["perplexity_comparison"] = compare_base_vs_finetuned(
            base_model_name=config["input"]["base_model"],
            finetuned_model_dir=MERGED_DIR,
            eval_dataset_name=config["awq"]["calibration_dataset"],
            num_samples=args.num_eval_samples,
            benchmark_runs=args.benchmark_runs,
        )
    except Exception as e:
        logger.warning(f"Perplexity comparison failed: {e}")
        report["perplexity_comparison"] = {"error": str(e)}

    # ---------- Part 2: Quantization size comparison ----------
    logger.info("=" * 60)
    logger.info("Comparing quantization artifacts (size)...")
    logger.info("=" * 60)

    report["quantization_comparison"] = compare_quantization_methods(config)

    # ---------- Save report ----------
    report_path = config["evaluation"]["report_output"]
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Evaluation report saved to: {report_path}")
    logger.info(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

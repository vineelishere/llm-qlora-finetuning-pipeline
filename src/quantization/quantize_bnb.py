import sys
import os
import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from config_loader import load_config
from logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="BitsAndBytes Quantization")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/quantization.yaml",
        help="Path to quantization config YAML file",
    )
    return parser.parse_args()


def merge_lora_with_base(base_model_name: str, lora_adapter_path: str):
    logger.info(f"Loading base model in full precision: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path)

    logger.info(f"Loading LoRA adapter from: {lora_adapter_path}")
    model_with_lora = PeftModel.from_pretrained(base_model, lora_adapter_path)

    logger.info("Merging LoRA weights into base model...")
    merged_model = model_with_lora.merge_and_unload()

    logger.info("Merge complete. Model is now standalone (no LoRA dependency).")
    return merged_model, tokenizer


def quantize_to_bnb_4bit(merged_model_dir: str, output_dir: str, quant_cfg: dict):
    logger.info("Loading merged model in 4-bit (bitsandbytes) precision...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=getattr(
            torch, quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16")
        ),
    )

    start_time = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        merged_model_dir,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(merged_model_dir)

    load_time = time.time() - start_time
    logger.info(f"4-bit model loaded in {load_time:.2f} seconds.")

    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)

    config_note_path = os.path.join(output_dir, "quantization_info.txt")
    with open(config_note_path, "w") as f:
        f.write(f"This model uses on-the-fly bitsandbytes 4-bit quantization.\n")
        f.write(f"Quant type: {quant_cfg.get('bnb_4bit_quant_type', 'nf4')}\n")
        f.write(f"Compute dtype: {quant_cfg.get('bnb_4bit_compute_dtype', 'bfloat16')}\n")
        f.write(f"Source model dir: {merged_model_dir}\n")
        f.write(f"Load time: {load_time:.2f} seconds\n")

    logger.info(f"Quantization info saved to: {output_dir}")
    return model, tokenizer, load_time


def main():
    args = parse_args()
    config = load_config(args.config)

    base_model_name = config["input"]["base_model"]
    lora_adapter_path = config["input"]["finetuned_model_path"]

    if not config["bitsandbytes"].get("enabled", True):
        logger.info("BitsAndBytes quantization disabled in config. Skipping.")
        return

    # Step 1: Merge LoRA with base model
    merged_model, tokenizer = merge_lora_with_base(base_model_name, lora_adapter_path)

    merged_dir = "outputs/merged_model"
    os.makedirs(merged_dir, exist_ok=True)
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    logger.info(f"Merged model saved to: {merged_dir}")

    # Free up memory before reloading in quantized form
    del merged_model
    torch.cuda.empty_cache()

    # Step 2: Quantize merged model to 4-bit
    output_dir = config["bitsandbytes"]["output_dir"]
    quantize_to_bnb_4bit(merged_dir, output_dir, config["bitsandbytes"])

    logger.info("BitsAndBytes quantization pipeline complete!")


if __name__ == "__main__":
    main()
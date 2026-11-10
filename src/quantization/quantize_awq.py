import sys
import os
import argparse
import time

from datasets import load_dataset
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from config_loader import load_config
from logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="AWQ Quantization")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/quantization.yaml",
        help="Path to quantization config YAML file",
    )
    return parser.parse_args()


def load_calibration_data(dataset_name: str, num_samples: int) -> list:
    logger.info(f"Loading calibration dataset: {dataset_name} ({num_samples} samples)")

    raw_dataset = load_dataset(dataset_name, split="train")
    raw_dataset = raw_dataset.select(range(min(num_samples, len(raw_dataset))))

    calibration_texts = []
    for example in raw_dataset:
        instruction = example.get("instruction", "")
        response = example.get("response", "")
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
        calibration_texts.append(text)

    logger.info(f"Calibration data ready: {len(calibration_texts)} samples")
    return calibration_texts


def quantize_with_awq(merged_model_dir: str, output_dir: str, awq_cfg: dict):
    logger.info(f"Loading model for AWQ quantization: {merged_model_dir}")

    model = AutoAWQForCausalLM.from_pretrained(merged_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(merged_model_dir)

    quant_config = {
        "zero_point": awq_cfg.get("zero_point", True),
        "q_group_size": awq_cfg.get("q_group_size", 128),
        "w_bit": awq_cfg.get("w_bit", 4),
        "version": "GEMM",
    }

    calibration_data = load_calibration_data(
        awq_cfg["calibration_dataset"],
        awq_cfg.get("calibration_samples", 128),
    )

    logger.info(f"Running AWQ quantization ({quant_config['w_bit']}-bit, group size {quant_config['q_group_size']})...")
    start_time = time.time()

    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=calibration_data,
    )

    quant_time = time.time() - start_time
    logger.info(f"AWQ quantization complete in {quant_time:.2f} seconds.")

    os.makedirs(output_dir, exist_ok=True)
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"AWQ quantized model saved to: {output_dir}")


def main():
    args = parse_args()
    config = load_config(args.config)

    if not config["awq"].get("enabled", True):
        logger.info("AWQ quantization disabled in config. Skipping.")
        return

    merged_dir = "outputs/merged_model"
    output_dir = config["awq"]["output_dir"]

    quantize_with_awq(merged_dir, output_dir, config["awq"])

    logger.info("AWQ quantization pipeline complete!")


if __name__ == "__main__":
    main()

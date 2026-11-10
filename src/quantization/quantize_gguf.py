import sys
import os
import subprocess
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from config_loader import load_config
from logger import get_logger

logger = get_logger(__name__)

LLAMA_CPP_DIR = "llama.cpp"


def parse_args():
    parser = argparse.ArgumentParser(description="GGUF Quantization")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/quantization.yaml",
        help="Path to quantization config YAML file",
    )
    return parser.parse_args()


def setup_llama_cpp():
    if not os.path.exists(LLAMA_CPP_DIR):
        logger.info("llama.cpp not found locally. Cloning...")
        subprocess.run(
            ["git", "clone", "https://github.com/ggerganov/llama.cpp.git", LLAMA_CPP_DIR],
            check=True,
        )
        logger.info("llama.cpp cloned successfully.")
    else:
        logger.info("llama.cpp repo already exists. Skipping clone.")

    requirements_path = os.path.join(LLAMA_CPP_DIR, "requirements.txt")
    if os.path.exists(requirements_path):
        logger.info("Installing llama.cpp Python requirements...")
        subprocess.run(
            ["pip", "install", "-r", requirements_path],
            check=True,
        )


def convert_to_gguf_fp16(merged_model_dir: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    fp16_output_path = os.path.join(output_dir, "model-fp16.gguf")

    convert_script = os.path.join(LLAMA_CPP_DIR, "convert_hf_to_gguf.py")

    logger.info(f"Converting HF model to GGUF (fp16): {merged_model_dir}")
    subprocess.run(
        [
            "python", convert_script,
            merged_model_dir,
            "--outfile", fp16_output_path,
            "--outtype", "f16",
        ],
        check=True,
    )

    logger.info(f"FP16 GGUF file created: {fp16_output_path}")
    return fp16_output_path


def quantize_gguf_levels(fp16_gguf_path: str, output_dir: str, quant_levels: list):
    quantize_binary = os.path.join(LLAMA_CPP_DIR, "build", "bin", "llama-quantize")

    for level in quant_levels:
        output_path = os.path.join(output_dir, f"model-{level}.gguf")
        logger.info(f"Quantizing to {level.upper()}...")

        subprocess.run(
            [quantize_binary, fp16_gguf_path, output_path, level.upper()],
            check=True,
        )

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"{level.upper()} quantized model saved: {output_path} ({size_mb:.1f} MB)")


def main():
    args = parse_args()
    config = load_config(args.config)

    if not config["gguf"].get("enabled", True):
        logger.info("GGUF quantization disabled in config. Skipping.")
        return

    merged_dir = "outputs/merged_model"
    output_dir = config["gguf"]["output_dir"]
    quant_levels = config["gguf"]["quant_levels"]

    # Step 1: llama.cpp setup
    setup_llama_cpp()

    # Step 2: Convert to FP16 GGUF (intermediate format)
    fp16_path = convert_to_gguf_fp16(merged_dir, output_dir)

    # Step 3: Quantize to multiple levels
    quantize_gguf_levels(fp16_path, output_dir, quant_levels)

    logger.info("GGUF quantization pipeline complete!")


if __name__ == "__main__":
    main()



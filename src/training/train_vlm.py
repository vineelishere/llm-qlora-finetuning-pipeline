"""
VLM (vision-language) QLoRA fine-tuning entry point.

Mirrors train_llm.py but works with an AutoProcessor + multimodal batches, so
image tokens are kept while the language layers are adapted with LoRA.

Usage:
    python src/training/train_vlm.py --config configs/vlm_finetune.yaml
"""
import sys
import os
import argparse

import torch
from transformers import (
    AutoProcessor,
    TrainingArguments,
    Trainer,
)

try:
    from transformers import AutoModelForVision2Seq
except ImportError:  # transformers >= 5 renamed the vision-to-text auto class
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq

# Put the shared utils and the project root on the path so sibling modules import cleanly
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from config_loader import load_config
from logger import get_logger
from lora_config import build_bnb_config, prepare_model_for_lora
from data.dataset_loader import load_dataset_for_training

logger = get_logger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="VLM QLoRA Fine-Tuning")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/vlm_finetune.yaml",
        help="Path to training config YAML file",
    )
    return parser.parse_args()


def load_model_and_processor(config: dict):
    """
    Load the processor (tokenizer + image processor) and the 4-bit quantized VLM.

    Args:
        config: Full config dictionary loaded from YAML.

    Returns:
        (model, processor) tuple.
    """
    model_name = config["model"]["base_model"]
    logger.info(f"Loading base VLM: {model_name}")

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=config["model"].get("trust_remote_code", True),
    )

    bnb_config = build_bnb_config(config["quantization"])

    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=config["model"].get("trust_remote_code", True),
        use_cache=config["model"].get("use_cache", False),
    )

    logger.info("Base VLM loaded successfully in 4-bit precision.")
    return model, processor


class VLMDataCollator:
    """Stack already-processed multimodal examples (pixel values + text tensors) into batches."""

    def __call__(self, features):
        batch = {}
        keys = features[0].keys()
        for key in keys:
            batch[key] = torch.stack([torch.as_tensor(f[key]) for f in features])
        return batch


def main():
    args = parse_args()

    logger.info(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # ---------- Step 1: Load model + processor ----------
    model, processor = load_model_and_processor(config)

    # ---------- Step 2: Apply LoRA ----------
    logger.info("Applying LoRA adapters to the model...")
    model = prepare_model_for_lora(model, config["lora"])

    # ---------- Step 3: Load the dataset ----------
    logger.info("Loading and processing image-text dataset...")
    train_dataset, val_dataset = load_dataset_for_training(
        config["dataset"], processor, model_type="vlm"
    )

    # ---------- Step 4: Weights & Biases init (optional) ----------
    if config["training"].get("report_to") == "wandb":
        try:
            import wandb
            wandb.init(
                project=config["wandb"]["project"],
                name=config["wandb"]["run_name"],
                config=config,
            )
            logger.info("W&B tracking initialized.")
        except Exception as e:
            logger.warning(f"W&B init failed, continuing without tracking: {e}")
            config["training"]["report_to"] = "none"

    # ---------- Step 5: Training arguments ----------
    train_cfg = config["training"]
    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        logging_steps=train_cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        optim=train_cfg["optim"],
        bf16=train_cfg.get("bf16", True),
        fp16=train_cfg.get("fp16", False),
        seed=train_cfg["seed"],
        report_to=train_cfg.get("report_to", "none"),
        remove_unused_columns=False,  # Required for VLMs: image columns must survive
    )

    # ---------- Step 6: Data collator ----------
    data_collator = VLMDataCollator()

    # ---------- Step 7: Build the trainer and train ----------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    logger.info("Starting VLM training...")
    trainer.train()
    logger.info("Training complete!")

    # ---------- Step 8: Save the final adapter ----------
    final_path = os.path.join(train_cfg["output_dir"], "final_model")
    trainer.model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    logger.info(f"Fine-tuned VLM (LoRA adapters) saved to: {final_path}")


if __name__ == "__main__":
    main()

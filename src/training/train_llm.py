"""
LLM QLoRA fine-tuning entry point.

Loads the configured base model in 4-bit, attaches LoRA adapters, tokenizes the
instruction dataset and trains with HuggingFace Trainer. All hyperparameters
come from the YAML config passed with --config.

Usage:
    python src/training/train_llm.py --config configs/llm_finetune.yaml
"""
import sys
import os
import argparse

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from config_loader import load_config
from logger import get_logger
from lora_config import build_bnb_config, prepare_model_for_lora
from data.dataset_loader import load_dataset_for_training

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="LLM QLoRA Fine-Tuning")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/llm_finetune.yaml",
        help="Path to training config YAML file",
    )
    return parser.parse_args()


def load_model_and_tokenizer(config: dict):
    model_name = config["model"]["base_model"]
    logger.info(f"Loading base model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=config["model"].get("trust_remote_code", True),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = build_bnb_config(config["quantization"])

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=config["model"].get("trust_remote_code", True),
        use_cache=config["model"].get("use_cache", False),
    )

    logger.info("Base model loaded successfully in 4-bit precision.")
    return model, tokenizer


def main():
    args = parse_args()

    logger.info(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # ---------- Step 1: Load model and tokenizer ----------
    model, tokenizer = load_model_and_tokenizer(config)

    # ---------- Step 2: Apply LoRA ----------
    logger.info("Applying LoRA adapters to the model...")
    model = prepare_model_for_lora(model, config["lora"])

    # ---------- Step 3: Load and tokenize the dataset ----------
    logger.info("Loading and tokenizing dataset...")
    train_dataset, val_dataset = load_dataset_for_training(
        config["dataset"], tokenizer, model_type="llm"
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
    )

    # ---------- Step 6: Data collator ----------
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ---------- Step 7: Build the trainer and train ----------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete!")

    # ---------- Step 8: Save the final adapter ----------
    final_path = os.path.join(train_cfg["output_dir"], "final_model")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Fine-tuned model (LoRA adapters) saved to: {final_path}")


if __name__ == "__main__":
    main()

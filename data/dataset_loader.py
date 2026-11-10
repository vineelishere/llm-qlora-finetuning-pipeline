from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoProcessor
from typing import Dict, Any, Tuple
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "utils"))
from logger import get_logger

logger = get_logger(__name__)


# ==========================================================
# LLM Dataset Handling
# ==========================================================

def format_llm_prompt(example: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Turn one instruction/response record into the prompt the model is trained on."""
    instruction = example.get(config["text_field"], "")
    response = example.get(config["response_field"], "")
    context = example.get(config.get("context_field", ""), "")

    if context and context.strip():
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Context:\n{context}\n\n"
            f"### Response:\n{response}"
        )
    else:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{response}"
        )

    return prompt


def load_llm_dataset(config: Dict[str, Any], tokenizer: AutoTokenizer) -> Tuple[Dataset, Dataset]:
    logger.info(f"Loading LLM dataset: {config['name']}")

    raw_dataset = load_dataset(config["name"], split=config["split"])
    logger.info(f"Dataset loaded. Total examples: {len(raw_dataset)}")

    max_samples = config.get("max_train_samples", None)
    if max_samples and max_samples < len(raw_dataset):
        raw_dataset = raw_dataset.select(range(max_samples))
        logger.info(f"Dataset trimmed to {max_samples} examples (for faster demo training).")

    def format_and_tokenize(example):
        text = format_llm_prompt(example, config)
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=config["max_seq_length"],
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    processed_dataset = raw_dataset.map(
        format_and_tokenize,
        remove_columns=raw_dataset.column_names,
        desc="Formatting and tokenizing LLM dataset",
    )

    # Train/val split
    val_ratio = config.get("val_split_ratio", 0.05)
    split_dataset = processed_dataset.train_test_split(test_size=val_ratio, seed=42)

    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    logger.info(f"Train examples: {len(train_dataset)} | Val examples: {len(val_dataset)}")

    return train_dataset, val_dataset


# ==========================================================
# VLM Dataset Handling
# ==========================================================

def format_vlm_conversation(example: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a Cauldron-style record into a single image + Q/A conversation."""
    images = example.get(config["image_field"], [])
    texts = example.get(config["conversation_field"], [])

    # One image and the first conversation turn keep training batches simple
    image = images[0] if images else None
    first_turn = texts[0] if texts else {"user": "", "assistant": ""}

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": first_turn.get("user", "")},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": first_turn.get("assistant", "")},
            ],
        },
    ]

    return {"image": image, "conversation": conversation}


def load_vlm_dataset(config: Dict[str, Any], processor: AutoProcessor) -> Tuple[Dataset, Dataset]:
    logger.info(f"Loading VLM dataset: {config['name']} (subset: {config.get('subset', 'default')})")

    if config.get("subset"):
        raw_dataset = load_dataset(config["name"], config["subset"], split=config["split"])
    else:
        raw_dataset = load_dataset(config["name"], split=config["split"])

    logger.info(f"Dataset loaded. Total examples: {len(raw_dataset)}")

    def process_example(example):
        formatted = format_vlm_conversation(example, config)

        # apply_chat_template uses the model-specific chat format from the processor
        text_prompt = processor.apply_chat_template(
            formatted["conversation"], tokenize=False, add_generation_prompt=False
        )

        inputs = processor(
            text=text_prompt,
            images=formatted["image"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=config["max_seq_length"],
        )

        # datasets.map works on a single example, so drop the batch dimension
        result = {k: v.squeeze(0) for k, v in inputs.items()}
        result["labels"] = result["input_ids"].clone()
        return result

    processed_dataset = raw_dataset.map(
        process_example,
        remove_columns=raw_dataset.column_names,
        desc="Formatting and processing VLM dataset",
    )

    val_ratio = config.get("val_split_ratio", 0.05)
    split_dataset = processed_dataset.train_test_split(test_size=val_ratio, seed=42)

    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    logger.info(f"Train examples: {len(train_dataset)} | Val examples: {len(val_dataset)}")

    return train_dataset, val_dataset


# ==========================================================
# Unified Entry Point
# ==========================================================

def load_dataset_for_training(
    config: Dict[str, Any],
    tokenizer_or_processor: Any,
    model_type: str = "llm",
) -> Tuple[Dataset, Dataset]:
    if model_type == "llm":
        return load_llm_dataset(config, tokenizer_or_processor)
    elif model_type == "vlm":
        return load_vlm_dataset(config, tokenizer_or_processor)
    else:
        raise ValueError(f"Invalid model_type: '{model_type}'. Expected 'llm' or 'vlm'.")


if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "utils"))
    from config_loader import load_config

    test_config = load_config("configs/llm_finetune.yaml")
    test_tokenizer = AutoTokenizer.from_pretrained(test_config["model"]["base_model"])

    train_ds, val_ds = load_dataset_for_training(
        test_config["dataset"], test_tokenizer, model_type="llm"
    )
    print(f"Test successful! Train size: {len(train_ds)}, Val size: {len(val_ds)}")

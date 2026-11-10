import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Dict, Any


def build_bnb_config(quant_config: Dict[str, Any]) -> BitsAndBytesConfig:
    compute_dtype = getattr(torch, quant_config.get("bnb_4bit_compute_dtype", "bfloat16"))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_config.get("load_in_4bit", True),
        bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_config.get("bnb_4bit_use_double_quant", True),
    )

    return bnb_config


def build_lora_config(lora_config: Dict[str, Any]) -> LoraConfig:

    return LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("lora_alpha", 32),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        bias=lora_config.get("bias", "none"),
        task_type=lora_config.get("task_type", "CAUSAL_LM"),
        target_modules=lora_config.get("target_modules", None),
    )


def prepare_model_for_lora(model, lora_config: Dict[str, Any]):
    model = prepare_model_for_kbit_training(model)

    peft_config = build_lora_config(lora_config)
    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    return model
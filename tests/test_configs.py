"""
Config validation tests.

These run without torch/transformers, so they are safe for CI on any machine and
catch the most common failure mode of this pipeline: a config that cannot be
parsed or that is missing a key a downstream script expects.
"""

import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "utils"))

from config_loader import load_config  # noqa: E402

CONFIG_DIR = PROJECT_ROOT / "configs"

REQUIRED_KEYS = {
    "llm_finetune.yaml": ["model", "quantization", "lora", "dataset", "training"],
    "vlm_finetune.yaml": ["model", "quantization", "lora", "dataset", "training"],
    "quantization.yaml": ["input", "bitsandbytes", "gguf", "awq", "evaluation"],
}


def test_config_files_exist():
    for name in REQUIRED_KEYS:
        assert (CONFIG_DIR / name).is_file(), f"missing config: {name}"


@pytest.mark.parametrize("config_name", sorted(REQUIRED_KEYS))
def test_config_parses_and_has_required_sections(config_name):
    config = load_config(str(CONFIG_DIR / config_name))

    for key in REQUIRED_KEYS[config_name]:
        assert key in config, f"{config_name} is missing required section '{key}'"


@pytest.mark.parametrize("config_name", ["llm_finetune.yaml", "vlm_finetune.yaml"])
def test_training_config_values_are_sane(config_name):
    config = load_config(str(CONFIG_DIR / config_name))
    train = config["training"]

    assert train["learning_rate"] > 0
    assert train["per_device_train_batch_size"] >= 1
    assert train["gradient_accumulation_steps"] >= 1
    assert train["num_train_epochs"] > 0
    assert config["lora"]["r"] > 0
    assert config["lora"]["task_type"] == "CAUSAL_LM"
    # Exactly one mixed-precision mode should be active
    assert not (train.get("bf16") and train.get("fp16"))


def test_quantization_levels_are_supported():
    config = load_config(str(CONFIG_DIR / "quantization.yaml"))
    supported = {"q4_k_m", "q5_k_m", "q8_0"}

    levels = set(config["gguf"]["quant_levels"])
    assert levels, "at least one GGUF quant level must be configured"
    assert levels <= supported, f"unsupported GGUF levels: {levels - supported}"
    assert config["input"]["base_model"]


def test_missing_config_raises_filenotfound():
    with pytest.raises(FileNotFoundError):
        load_config(str(CONFIG_DIR / "does_not_exist.yaml"))


def test_non_mapping_config_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a\n- list\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(str(bad))


def test_llm_prompt_formatting_uses_context_when_present():
    # Imported lazily: this module pulls in pandas/HF only when present
    pytest.importorskip("transformers")
    pytest.importorskip("datasets")
    sys.path.insert(0, str(PROJECT_ROOT))
    from data.dataset_loader import format_llm_prompt

    config = {"text_field": "instruction", "response_field": "response", "context_field": "context"}

    with_context = format_llm_prompt(
        {"instruction": "Summarize", "response": "Done", "context": "Some text"}, config
    )
    assert "### Context:" in with_context
    assert with_context.endswith("Done")

    without_context = format_llm_prompt(
        {"instruction": "Say hi", "response": "Hi", "context": ""}, config
    )
    assert "### Context:" not in without_context
    assert "### Instruction:\nSay hi" in without_context

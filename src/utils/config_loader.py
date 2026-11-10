"""
Config Loader Utility
---------------------
Single place where YAML config files are read into plain Python dictionaries.
Every training / quantization / evaluation script goes through this module so
config handling stays in one place (DRY).
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Read a YAML config file and return it as a dictionary.

    Args:
        config_path: Path to the YAML file (e.g. "configs/llm_finetune.yaml").

    Returns:
        Dictionary holding every value from the config.

    Raises:
        FileNotFoundError: if the config file does not exist.
        yaml.YAMLError: if the YAML has a syntax error.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"YAML parsing error in {config_path}: {e}")

    if not isinstance(config, dict):
        raise ValueError(f"Config file {config_path} must contain a YAML mapping at the top level.")

    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Write a config dictionary back to disk as YAML.

    Useful when a pipeline step needs to record the exact config it ran with
    next to its outputs.

    Args:
        config: Dictionary to persist.
        output_path: Destination YAML path (parent dirs are created).
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    # Quick check: run this file directly to confirm a config loads.
    test_config = load_config("configs/llm_finetune.yaml")
    print("Config loaded successfully:")
    print(test_config)

"""HuggingFace model download utilities."""

import json
from pathlib import Path
from typing import Tuple

from huggingface_hub import hf_hub_download

# HuggingFace repo and model files
HF_REPO = "yuliangguo/depth-any-camera"

MODELS = {
    "outdoor-resnet101": {
        "config": "dac_resnet101_outdoor.json",
        "weights": "dac_resnet101_outdoor.pt",
    },
    "outdoor-swinl": {
        "config": "dac_swinl_outdoor.json",
        "weights": "dac_swinl_outdoor.pt",
    },
    "indoor-resnet101": {
        "config": "dac_resnet101_indoor.json",
        "weights": "dac_resnet101_indoor.pt",
    },
    "indoor-swinl": {
        "config": "dac_swinl_indoor.json",
        "weights": "dac_swinl_indoor.pt",
    },
}


def download_model(name: str) -> Tuple[Path, Path]:
    """Download model config and weights from HuggingFace.
    
    Files are cached locally after first download.
    
    Args:
        name: Model name (e.g., 'outdoor-resnet101')
    
    Returns:
        Tuple of (config_path, weights_path)
    
    Raises:
        ValueError: If model name is not recognized
    """
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODELS.keys())}")
    
    model_info = MODELS[name]
    
    config_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=model_info["config"],
    )
    
    weights_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=model_info["weights"],
    )
    
    return Path(config_path), Path(weights_path)


def load_config(config_path: Path) -> dict:
    """Load model configuration from JSON file.
    
    Args:
        config_path: Path to config JSON file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        return json.load(f)

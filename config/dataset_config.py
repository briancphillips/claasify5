"""Dataset configuration for the experiment framework.

This module defines dataset-specific configurations and defaults.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from .base_config import BaseConfig


@dataclass
class DataAugConfig(BaseConfig):
    """Data augmentation configuration."""

    random_crop: bool = True
    random_crop_padding: int = 4
    random_horizontal_flip: bool = True
    normalize: bool = True
    normalize_mean: List[float] = field(
        default_factory=lambda: [0.5071, 0.4867, 0.4408]
    )
    normalize_std: List[float] = field(default_factory=lambda: [0.2675, 0.2565, 0.2761])


@dataclass
class DatasetConfig(BaseConfig):
    """Dataset configuration."""

    name: str
    subset_size: Optional[int] = None
    batch_size: int = 128
    augmentation: DataAugConfig = field(default_factory=DataAugConfig)
    data_dir: str = "data"


# Dataset-specific configurations
CIFAR100_CONFIG = {
    "name": "cifar100",
    "subset_size": 50000,
    "batch_size": 128,
    "augmentation": {
        "normalize_mean": [0.5071, 0.4867, 0.4408],
        "normalize_std": [0.2675, 0.2565, 0.2761],
    },
}

GTSRB_CONFIG = {
    "name": "gtsrb",
    "subset_size": 39209,
    "batch_size": 128,
    "augmentation": {
        "normalize_mean": [0.3337, 0.3064, 0.3171],
        "normalize_std": [0.2672, 0.2564, 0.2629],
    },
}

IMAGENETTE_CONFIG = {
    "name": "imagenette",
    "subset_size": 9469,
    "batch_size": 64,
    "augmentation": {
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
    },
}

DATASET_CONFIGS = {
    "cifar100": CIFAR100_CONFIG,
    "gtsrb": GTSRB_CONFIG,
    "imagenette": IMAGENETTE_CONFIG,
}


def create_dataset_config(dataset_name: str, **overrides) -> DatasetConfig:
    """Create dataset configuration with optional overrides.

    Args:
        dataset_name: Name of the dataset
        **overrides: Additional configuration overrides

    Returns:
        DatasetConfig: Dataset configuration
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Start with default configuration
    config_dict = DATASET_CONFIGS[dataset_name].copy()

    # Apply overrides
    for key, value in overrides.items():
        if key in config_dict:
            if isinstance(value, dict) and isinstance(config_dict[key], dict):
                config_dict[key].update(value)
            else:
                config_dict[key] = value

    return DatasetConfig(**config_dict)

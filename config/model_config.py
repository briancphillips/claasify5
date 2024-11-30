"""Model configuration for the experiment framework.

This module defines model architectures and their configurations.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from .base_config import BaseConfig


@dataclass
class ModelConfig(BaseConfig):
    """Model architecture configuration."""

    name: str
    num_classes: int
    pretrained: bool = False

    # Architecture-specific parameters
    depth: Optional[int] = None
    widen_factor: Optional[int] = None
    dropout_rate: Optional[float] = None

    def __post_init__(self):
        """Validate model configuration."""
        if self.name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model architecture: {self.name}")


@dataclass
class TrainingConfig(BaseConfig):
    """Model training configuration."""

    epochs: int = 200
    batch_size: int = 128
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.0005
    lr_schedule: List[int] = field(default_factory=lambda: [60, 120, 160])
    lr_factor: float = 0.2

    # Advanced training options
    use_swa: bool = False  # Stochastic Weight Averaging
    swa_start: int = 160
    swa_lr: float = 0.05

    grad_clip: Optional[float] = None
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2


# Model-specific configurations
WIDERESNET_CONFIG = {
    "name": "wrn-28-10",
    "depth": 28,
    "widen_factor": 10,
    "dropout_rate": 0.3,
    "num_classes": 100,  # Will be overridden by dataset
    "pretrained": False,
}

CUSTOM_CNN_CONFIG = {
    "name": "custom-cnn",
    "num_classes": 43,  # Will be overridden by dataset
    "pretrained": False,
}

RESNET50_CONFIG = {
    "name": "resnet50",
    "num_classes": 10,  # Will be overridden by dataset
    "pretrained": True,
}

MODEL_CONFIGS = {
    "wrn-28-10": WIDERESNET_CONFIG,
    "custom-cnn": CUSTOM_CNN_CONFIG,
    "resnet50": RESNET50_CONFIG,
}

# Dataset to model mapping
DATASET_MODEL_MAPPING = {
    "cifar100": "wrn-28-10",
    "gtsrb": "custom-cnn",
    "imagenette": "resnet50",
}

# Dataset-specific training configurations
DATASET_TRAINING_CONFIGS = {
    "cifar100": {
        "epochs": 200,
        "batch_size": 128,
        "learning_rate": 0.1,
        "use_swa": True,
    },
    "gtsrb": {
        "epochs": 100,
        "batch_size": 128,
        "learning_rate": 0.01,
        "use_swa": False,
    },
    "imagenette": {
        "epochs": 100,
        "batch_size": 64,
        "learning_rate": 0.001,
        "use_swa": False,
    },
}


def create_model_config(
    dataset_name: str, model_name: Optional[str] = None, **overrides
) -> tuple[ModelConfig, TrainingConfig]:
    """Create model and training configurations for a dataset.

    Args:
        dataset_name: Name of the dataset
        model_name: Optional model name (if not specified, uses dataset default)
        **overrides: Additional configuration overrides

    Returns:
        tuple[ModelConfig, TrainingConfig]: Model and training configurations
    """
    # Determine model name
    if model_name is None:
        if dataset_name not in DATASET_MODEL_MAPPING:
            raise ValueError(f"No default model for dataset: {dataset_name}")
        model_name = DATASET_MODEL_MAPPING[dataset_name]

    # Get base model config
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model architecture: {model_name}")
    model_config_dict = MODEL_CONFIGS[model_name].copy()

    # Get dataset-specific training config
    training_config_dict = DATASET_TRAINING_CONFIGS.get(dataset_name, {}).copy()

    # Apply overrides
    model_overrides = overrides.get("model", {})
    training_overrides = overrides.get("training", {})

    model_config_dict.update(model_overrides)
    training_config_dict.update(training_overrides)

    return ModelConfig(**model_config_dict), TrainingConfig(**training_config_dict)

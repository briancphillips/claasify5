"""Poisoning attack configuration for the experiment framework.

This module defines configurations for different poisoning attacks.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum, auto
from .base_config import BaseConfig


class PoisonType(str, Enum):
    """Types of poisoning attacks."""

    PGD = "pgd"
    GRADIENT_ASCENT = "gradient_ascent"
    LABEL_FLIP = "label_flip"
    NONE = "none"


@dataclass
class PGDConfig(BaseConfig):
    """Projected Gradient Descent attack configuration."""

    eps: float = 0.3
    alpha: float = 0.01
    steps: int = 40
    random_start: bool = True
    targeted: bool = False


@dataclass
class GradientAscentConfig(BaseConfig):
    """Gradient Ascent attack configuration."""

    steps: int = 50
    iterations: int = 100
    learning_rate: float = 0.1
    momentum: float = 0.9
    targeted: bool = False


@dataclass
class LabelFlipConfig(BaseConfig):
    """Label flipping attack configuration."""

    source_class: Optional[int] = None
    target_class: Optional[int] = None
    random_flip: bool = True


@dataclass
class PoisonConfig(BaseConfig):
    """Main poisoning configuration."""

    poison_type: PoisonType = PoisonType.NONE
    poison_ratio: float = 0.1
    batch_size: int = 32
    random_seed: int = 42

    # Attack-specific configurations
    pgd: PGDConfig = field(default_factory=PGDConfig)
    gradient_ascent: GradientAscentConfig = field(default_factory=GradientAscentConfig)
    label_flip: LabelFlipConfig = field(default_factory=LabelFlipConfig)

    def __post_init__(self):
        """Convert string poison type to enum if necessary."""
        if isinstance(self.poison_type, str):
            self.poison_type = PoisonType(self.poison_type.lower())


# Default configurations for each attack type
DEFAULT_PGD_CONFIG = {
    "eps": 0.3,
    "alpha": 0.01,
    "steps": 40,
    "random_start": True,
    "targeted": False,
}

DEFAULT_GRADIENT_ASCENT_CONFIG = {
    "steps": 50,
    "iterations": 100,
    "learning_rate": 0.1,
    "momentum": 0.9,
    "targeted": False,
}

DEFAULT_LABEL_FLIP_CONFIG = {"random_flip": True}

# Dataset-specific poison configurations
DATASET_POISON_CONFIGS = {
    "cifar100": {
        "poison_ratio": 0.1,
        "batch_size": 32,
        "pgd": {"eps": 0.3, "steps": 40},
        "gradient_ascent": {"steps": 50, "iterations": 100},
        "label_flip": {"random_flip": True},
    },
    "gtsrb": {
        "poison_ratio": 0.05,
        "batch_size": 32,
        "pgd": {"eps": 0.2, "steps": 30},
        "gradient_ascent": {"steps": 40, "iterations": 80},
        "label_flip": {"random_flip": True},
    },
    "imagenette": {
        "poison_ratio": 0.05,
        "batch_size": 32,
        "pgd": {"eps": 0.2, "steps": 30},
        "gradient_ascent": {"steps": 40, "iterations": 80},
        "label_flip": {"random_flip": True},
    },
}


def create_poison_config(
    dataset_name: str, poison_type: str = "none", **overrides
) -> PoisonConfig:
    """Create poisoning configuration for a dataset.

    Args:
        dataset_name: Name of the dataset
        poison_type: Type of poisoning attack
        **overrides: Additional configuration overrides

    Returns:
        PoisonConfig: Poisoning configuration
    """
    # Get dataset-specific config
    dataset_config = DATASET_POISON_CONFIGS.get(dataset_name, {}).copy()

    # Create base config
    config_dict = {
        "poison_type": poison_type,
        "poison_ratio": dataset_config.get("poison_ratio", 0.1),
        "batch_size": dataset_config.get("batch_size", 32),
        "random_seed": 42,
    }

    # Add attack-specific config if applicable
    if poison_type in ["pgd", "gradient_ascent", "label_flip"]:
        attack_config = dataset_config.get(poison_type, {}).copy()
        config_dict[poison_type] = attack_config

    # Apply overrides
    for key, value in overrides.items():
        if (
            isinstance(value, dict)
            and key in config_dict
            and isinstance(config_dict[key], dict)
        ):
            config_dict[key].update(value)
        else:
            config_dict[key] = value

    return PoisonConfig(**config_dict)

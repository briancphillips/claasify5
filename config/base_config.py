"""Base configuration classes for the experiment framework.

This module defines the core configuration classes that all other configurations
will inherit from or use. It provides a consistent interface for configuration
management across the entire framework.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from pathlib import Path
import yaml


@dataclass
class BaseConfig:
    """Base configuration class with common functionality."""

    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""

        def _recursive_update(obj, updates):
            for key, value in updates.items():
                if hasattr(obj, key):
                    if isinstance(value, dict) and hasattr(
                        getattr(obj, key), "__dataclass_fields__"
                    ):
                        _recursive_update(getattr(obj, key), value)
                    else:
                        setattr(obj, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "BaseConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def save_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


@dataclass
class HardwareConfig(BaseConfig):
    """Hardware-specific configuration."""

    device: str = "auto"  # auto, cuda, mps, or cpu
    gpu_ids: list[int] = field(default_factory=lambda: [0])
    num_workers: int = 4
    pin_memory: bool = True
    use_amp: bool = True  # Automatic Mixed Precision

    def __post_init__(self):
        """Validate and adjust hardware configuration."""
        import torch

        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
                self.num_workers = 0  # MPS doesn't support multiprocessing
            else:
                self.device = "cpu"


@dataclass
class LoggingConfig(BaseConfig):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_dir: str = "logs"
    save_frequency: int = 10

    def __post_init__(self):
        """Create log directory if it doesn't exist."""
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class OutputConfig(BaseConfig):
    """Output configuration for experiment results."""

    base_dir: str = "results"
    save_models: bool = True
    consolidated_file: str = "all_results.csv"
    save_individual_results: bool = True
    checkpoint_dir: str = "checkpoints"

    def __post_init__(self):
        """Create output directories if they don't exist."""
        Path(self.base_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class ExperimentTypeConfig(BaseConfig):
    """Configuration for experiment type and basic parameters."""

    type: str = "classification"  # classification, poison, or traditional
    name: str = "default"
    description: str = ""
    seed: int = 42
    debug: bool = False

"""Experiment configuration utilities."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    name: str
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    """Model configuration."""

    name: Optional[str] = None
    checkpoint_path: Optional[str] = None


@dataclass
class HardwareConfig:
    """Hardware configuration."""

    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ExperimentConfig:
    """Experiment configuration."""

    dataset: DatasetConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    overrides: Dict[str, Any] = field(default_factory=dict)


def create_experiment_config(
    dataset_name: str,
    model_name: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    overrides: Optional[Dict[str, Any]] = None,
) -> ExperimentConfig:
    """Create experiment configuration."""
    dataset_config = DatasetConfig(
        name=dataset_name,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model_config = ModelConfig(
        name=model_name,
        checkpoint_path=checkpoint_path,
    )

    hardware_config = HardwareConfig(
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return ExperimentConfig(
        dataset=dataset_config,
        model=model_config,
        hardware=hardware_config,
        overrides=overrides or {},
    )

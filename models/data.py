"""Dataset loading and preprocessing utilities."""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from utils.logging import get_logger

logger = get_logger(__name__)


def get_dataset_stats(dataset_name: str) -> Tuple[list, list]:
    """Get normalization stats for dataset."""
    if dataset_name == "cifar100":
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    elif dataset_name == "gtsrb":
        mean = [0.3337, 0.3064, 0.3171]
        std = [0.2672, 0.2564, 0.2629]
    elif dataset_name == "imagenette":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return mean, std


def get_transforms(dataset_name: str, train: bool = True) -> transforms.Compose:
    """Get transforms for dataset."""
    mean, std = get_dataset_stats(dataset_name)

    if train:
        if dataset_name == "cifar100":
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        elif dataset_name == "gtsrb":
            transform = transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.RandomRotation(15),
                    transforms.RandomAffine(0, translate=(0.1, 0.1)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        elif dataset_name == "imagenette":
            transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
    else:
        if dataset_name == "cifar100":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        elif dataset_name == "gtsrb":
            transform = transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        elif dataset_name == "imagenette":
            transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )

    return transform


def get_dataset(
    dataset_name: str,
    train: bool = True,
    subset_size: Optional[int] = None,
    transform: Optional[transforms.Compose] = None,
) -> Dataset:
    """Get dataset with optional subset."""
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Get default transforms if none provided
    if transform is None:
        transform = get_transforms(dataset_name, train)

    # Load dataset
    if dataset_name == "cifar100":
        dataset = torchvision.datasets.CIFAR100(
            root=str(data_dir), train=train, download=True, transform=transform
        )
    elif dataset_name == "gtsrb":
        if train:
            dataset = torchvision.datasets.GTSRB(
                root=str(data_dir), split="train", download=True, transform=transform
            )
        else:
            dataset = torchvision.datasets.GTSRB(
                root=str(data_dir), split="test", download=True, transform=transform
            )
    elif dataset_name == "imagenette":
        try:
            # Try loading without download first
            dataset = torchvision.datasets.Imagenette(
                root=str(data_dir),
                split="train" if train else "val",
                download=False,
                transform=transform,
            )
            logger.info("Successfully loaded existing ImageNette dataset")
        except RuntimeError:
            # If that fails, try downloading
            logger.info("Downloading ImageNette dataset...")
            dataset = torchvision.datasets.Imagenette(
                root=str(data_dir),
                split="train" if train else "val",
                download=True,
                transform=transform,
            )
            logger.info("ImageNette dataset downloaded and loaded")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Create subset if specified
    if subset_size is not None:
        if subset_size > len(dataset):
            logger.warning(
                f"Requested subset size {subset_size} is larger than dataset size {len(dataset)}. "
                "Using full dataset."
            )
            return dataset

        indices = torch.randperm(len(dataset))[:subset_size]
        dataset = Subset(dataset, indices)

    return dataset

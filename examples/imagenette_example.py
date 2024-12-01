"""Example script for running ImageNette experiments."""

import logging
from pathlib import Path
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

from config.experiment_config import create_experiment_config
from experiments.traditional import TraditionalExperiment
from models.data import get_dataset
from models.factory import get_model
from utils.logging import setup_logging, get_logger

# Set up logging first thing
setup_logging(level=logging.INFO)
logger = get_logger(__name__)

# Check for existing checkpoint
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "imagenette_best.pt"
if CHECKPOINT_PATH.exists():
    logger.info(f"Found checkpoint at {CHECKPOINT_PATH}")
else:
    logger.warning("No checkpoint found, will use untrained model")
    CHECKPOINT_PATH = None


def run_imagenette_experiment(subset_size=None):
    """Run ImageNette experiment."""
    logger.info(f"Starting ImageNette experiment (subset_size={subset_size})")
    logger.info("Python process is running...")

    try:
        # Create experiment with proper model name
        logger.info("Creating experiment configuration...")
        imagenette_exp = TraditionalExperiment(
            config=create_experiment_config(
                dataset_name="imagenette",
                model_name="imagenette-net",  # This is the model we want to use
                checkpoint_path=CHECKPOINT_PATH,  # Use checkpoint if available
                batch_size=16,  # Smaller batch size for memory (224x224 images)
            ),
            subset_size=subset_size,
        )

        # Load dataset
        logger.info("Loading ImageNette dataset...")
        train_dataset = get_dataset("imagenette", train=True)
        test_dataset = get_dataset("imagenette", train=False)

        # If using subset, take first subset_size samples
        if subset_size:
            logger.info(f"Creating subset of {subset_size} samples...")
            train_dataset = torch.utils.data.Subset(
                train_dataset, range(min(subset_size, len(train_dataset)))
            )
            test_dataset = torch.utils.data.Subset(
                test_dataset, range(min(subset_size // 5, len(test_dataset)))
            )

        logger.info(
            f"Dataset sizes - Train: {len(train_dataset)}, Test: {len(test_dataset)}"
        )

        # Extract features
        logger.info("Extracting features from training set...")
        train_features, train_labels = imagenette_exp.extract_features(train_dataset)
        logger.info("Extracting features from test set...")
        test_features, test_labels = imagenette_exp.extract_features(test_dataset)

        # Train and evaluate classifiers
        results = {}
        for clf_name in imagenette_exp.classifiers:
            logger.info(f"Training {clf_name} classifier...")
            train_time, inference_time, accuracy = imagenette_exp.train_and_evaluate(
                clf_name, train_features, train_labels, test_features, test_labels
            )
            results[clf_name] = {
                "accuracy": accuracy,
                "train_time": train_time,
                "inference_time": inference_time,
            }

        # Log results
        logger.info("\nImageNette Results:")
        for clf_name, metrics in results.items():
            logger.info(
                f"{clf_name}: {metrics['accuracy']:.2f}% "
                f"(Train: {metrics['train_time']:.2f}s, "
                f"Inference: {metrics['inference_time']:.2f}s)"
            )

        return True

    except Exception as e:
        logger.error(f"Error in ImageNette experiment: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        return False


def main():
    """Run example experiments."""
    # Run experiments with very small subset for testing
    if not run_imagenette_experiment(
        subset_size=500
    ):  # Just 500 samples for quick test
        sys.exit(1)


if __name__ == "__main__":
    main()

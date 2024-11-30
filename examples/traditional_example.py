"""Example script for running traditional classifier experiments."""

import logging
from pathlib import Path
import torch

from config.experiment_config import create_experiment_config
from experiments.traditional import TraditionalExperiment
from models.data import get_dataset
from utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def run_cifar100_experiment(subset_size=5000):
    """Run CIFAR-100 experiment."""
    logger.info(f"Step 1: Running CIFAR-100 experiment (subset_size={subset_size})")
    try:
        # Create experiment
        cifar_exp = TraditionalExperiment(
            config=create_experiment_config(
                dataset_name="cifar100",
                model_name="wrn-28-10",
                checkpoint_path="checkpoints/wideresnet/wideresnet_best.pt",
            ),
            subset_size=subset_size,
        )

        # Load dataset
        train_dataset = get_dataset("cifar100", train=True)
        test_dataset = get_dataset("cifar100", train=False)

        # If using subset, take first subset_size samples
        if subset_size:
            train_dataset = torch.utils.data.Subset(
                train_dataset, range(min(subset_size, len(train_dataset)))
            )
            test_dataset = torch.utils.data.Subset(
                test_dataset, range(min(subset_size // 5, len(test_dataset)))
            )

        # Extract features
        train_features, train_labels = cifar_exp.extract_features(train_dataset)
        test_features, test_labels = cifar_exp.extract_features(test_dataset)

        # Train and evaluate classifiers
        results = {}
        for clf_name in cifar_exp.classifiers:
            logger.info(f"Training {clf_name}")
            train_time, inference_time, accuracy = cifar_exp.train_and_evaluate(
                clf_name, train_features, train_labels, test_features, test_labels
            )
            results[clf_name] = {
                "accuracy": accuracy,
                "train_time": train_time,
                "inference_time": inference_time,
            }

        # Log results
        logger.info("\nCIFAR-100 results:")
        for clf_name, metrics in results.items():
            logger.info(
                f"{clf_name}: {metrics['accuracy']:.2f}% "
                f"(Train: {metrics['train_time']:.2f}s, "
                f"Inference: {metrics['inference_time']:.2f}s)"
            )

        # Check if results are good enough
        if all(metrics["accuracy"] < 10.0 for metrics in results.values()):
            logger.error(
                "CIFAR-100 experiment failed or returned poor results, stopping here."
            )
            return False

        return True

    except Exception as e:
        logger.error(f"Error in CIFAR-100 experiment: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        logger.error(
            "CIFAR-100 experiment failed or returned poor results, stopping here."
        )
        return False


def main():
    """Run example experiments."""
    # Set up logging
    setup_logging()

    # Create checkpoints directory
    Path("checkpoints").mkdir(exist_ok=True)

    # Run experiments with subset
    if not run_cifar100_experiment(subset_size=5000):
        return


if __name__ == "__main__":
    main()

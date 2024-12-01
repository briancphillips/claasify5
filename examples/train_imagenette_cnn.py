"""Script for training the ImageNette CNN model."""

import logging
from pathlib import Path
import sys
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.data import get_dataset
from models.factory import get_model
from utils.logging import setup_logging, get_logger
from utils.device import get_device

# Set up logging
setup_logging(level=logging.INFO)
logger = get_logger(__name__)

# Training configuration
config = {
    "batch_size": 32,
    "num_workers": 4,
    "pin_memory": True,
    "epochs": 20,
    "learning_rate": 0.001,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "label_smoothing": 0.1,
}


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc="Training")
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar.set_postfix(
            {
                "Loss": f"{running_loss/len(train_loader):.3f}",
                "Acc": f"{100.*correct/total:.2f}%",
            }
        )

    return running_loss / len(train_loader), 100.0 * correct / total


def evaluate(model, test_loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return running_loss / len(test_loader), 100.0 * correct / total


def main(subset_size=None, resume=False):
    """Train ImageNette CNN model."""
    # Setup device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Enable cuDNN benchmarking for better performance
    if device.type == "cuda":
        cudnn.benchmark = True

    # Create model
    model = get_model("imagenette").to(device)
    logger.info("Created ImageNette model")

    # Check for checkpoint
    checkpoint_path = Path("checkpoints/imagenette_best.pt")
    start_epoch = 1
    best_acc = 0.0

    if resume and checkpoint_path.exists():
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint["model_state_dict"])
                best_acc = checkpoint.get("best_acc", 0.0)
                start_epoch = checkpoint.get("epoch", 1) + 1
                logger.info(
                    f"Resumed from epoch {start_epoch} with best accuracy: {best_acc:.2f}%"
                )
            else:
                model.load_state_dict(checkpoint)
                logger.info("Loaded model weights from checkpoint")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            logger.error("Starting from scratch")
            start_epoch = 1
            best_acc = 0.0

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    optimizer = optim.SGD(
        model.parameters(),
        lr=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = get_dataset("imagenette", train=True)
    test_dataset = get_dataset("imagenette", train=False)

    # Create subsets if specified
    if subset_size:
        logger.info(f"Creating subsets of size {subset_size}...")
        train_indices = torch.randperm(len(train_dataset))[:subset_size]
        test_indices = torch.randperm(len(test_dataset))[: subset_size // 5]
        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(config["batch_size"], len(train_dataset)),
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=min(config["batch_size"], len(test_dataset)),
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )

    logger.info(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

    # Create checkpoints directory
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)

    # Training loop
    logger.info("Starting training...")
    start_time = time.time()

    for epoch in range(start_epoch, config["epochs"] + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Log metrics
        logger.info(
            f"\nEpoch {epoch}/{config['epochs']}:"
            f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%"
            f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
        )

        # Save best model
        if test_acc > best_acc:
            logger.info(f"New best accuracy: {test_acc:.2f}%")
            best_acc = test_acc
            # Save checkpoint with more information
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
                "train_acc": train_acc,
                "test_acc": test_acc,
            }
            torch.save(checkpoint, checkpoints_dir / "imagenette_best.pt")

    # Log final metrics
    total_time = time.time() - start_time
    logger.info(f"\nTraining completed in {total_time/60:.2f} minutes")
    logger.info(f"Best test accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Train ImageNette CNN model")
        parser.add_argument(
            "--subset-size",
            type=int,
            default=500,
            help="Number of training samples to use (default: 500)",
        )
        parser.add_argument(
            "--resume",
            action="store_true",
            help="Resume training from checkpoint if available",
        )
        args = parser.parse_args()

        main(subset_size=args.subset_size, resume=args.resume)
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        sys.exit(1)

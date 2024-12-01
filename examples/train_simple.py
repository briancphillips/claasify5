"""Simple training script for ImageNette dataset."""

import logging
import sys
from pathlib import Path
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms

from models.data import get_dataset
from models.factory import get_model
from utils.device import get_device

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def get_transforms():
    """Get ImageNette transforms."""
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, test_transform


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
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

        if (batch_idx + 1) % 2 == 0 or (batch_idx + 1) == len(train_loader):
            print(
                f"Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                f"Loss: {running_loss/(batch_idx+1):.3f} "
                f"Acc: {100.*correct/total:.1f}%"
            )

    return running_loss / len(train_loader), 100.0 * correct / total


def evaluate(model, val_loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return running_loss / len(val_loader), 100.0 * correct / total


def main(args):
    """Train model."""
    # Setup device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Create model
    model = get_model("imagenette").to(device)
    logger.info("Created ImageNette model")

    # Get transforms
    train_transform, test_transform = get_transforms()

    # Load datasets
    train_dataset = get_dataset("imagenette", train=True, transform=train_transform)
    val_dataset = get_dataset("imagenette", train=False, transform=test_transform)

    # Create subsets if specified
    if args.subset_size:
        train_indices = torch.randperm(len(train_dataset))[: args.subset_size]
        val_indices = torch.randperm(len(val_dataset))[: args.subset_size // 5]
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[60, 120, 160], gamma=0.2
    )

    # Setup checkpointing
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / "imagenette_best.pt"

    # Training loop
    logger.info("Starting training...")
    best_acc = 0.0
    try:
        for epoch in range(1, args.epochs + 1):
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )

            # Evaluate
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            print(
                f"Epoch {epoch} Summary - Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.1f}%, "
                f"Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.1f}%"
            )

            # Update learning rate
            scheduler.step()

            # Save best model
            if val_acc > best_acc:
                logger.info(f"New best accuracy: {val_acc:.2f}%")
                best_acc = val_acc
                torch.save(model.state_dict(), checkpoint_path)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

    logger.info(f"Training completed. Best accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ImageNette model")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs (default: 5)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--subset-size", type=int, help="Number of training samples to use (optional)"
    )

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        sys.exit(1)

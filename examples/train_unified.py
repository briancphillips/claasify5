"""Unified training script for ImageNette and GTSRB datasets."""

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

# Set up logging only once at the module level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def get_transforms(dataset_name):
    """Get dataset-specific transforms."""
    if dataset_name == "imagenette":
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    elif dataset_name == "cifar100":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
                ),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
                ),
            ]
        )
    elif dataset_name == "gtsrb":
        train_transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629]
                ),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629]
                ),
            ]
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_transform, test_transform


def setup_device():
    """Set up the appropriate device and training settings."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Enable cuDNN benchmarking for better performance on CUDA
        torch.backends.cudnn.benchmark = True
        use_amp = True  # Use mixed precision on CUDA
        scaler = torch.cuda.amp.GradScaler()
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        use_amp = False  # No mixed precision on MPS
        scaler = None
    else:
        device = torch.device("cpu")
        use_amp = False
        scaler = None

    logger.info(f"Using device: {device} (Mixed precision: {use_amp})")
    return device, use_amp, scaler


def train_epoch(model, train_loader, criterion, optimizer, device, scaler, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        if scaler is not None:  # Using mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:  # Standard precision
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


def evaluate(model, val_loader, criterion, device, scaler):
    """Evaluate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            if scaler is not None:  # Using mixed precision
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
            else:  # Standard precision
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return running_loss / len(val_loader), 100.0 * correct / total


def main(args):
    """Train model."""
    # Setup device and mixed precision
    device, use_amp, scaler = setup_device()

    # Create model and move to device
    model = get_model(args.dataset).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    logger.info(f"Created {args.dataset} model")

    # Get dataset-specific transforms
    train_transform, test_transform = get_transforms(args.dataset)

    # Load datasets with transforms
    train_dataset = get_dataset(args.dataset, train=True, transform=train_transform)
    val_dataset = get_dataset(args.dataset, train=False, transform=test_transform)

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
    checkpoint_path = checkpoint_dir / f"{args.dataset}_best.pt"

    # Load checkpoint if resuming
    start_epoch = 1
    best_acc = 0.0
    if args.resume and checkpoint_path.exists():
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_acc = checkpoint.get("best_acc", 0.0)
            logger.info(
                f"Resumed from epoch {start_epoch} with best accuracy: {best_acc:.2f}%"
            )

    # Training loop
    logger.info("Starting training...")
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, scaler, epoch
            )

            # Evaluate
            val_loss, val_acc = evaluate(model, val_loader, criterion, device, scaler)
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
                state = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_acc": best_acc,
                }
                torch.save(state, checkpoint_path)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

    logger.info(f"Training completed. Best accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["imagenette", "gtsrb", "cifar100"],
        help="Dataset to train on",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of epochs (default: 200)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Initial learning rate (default: 0.1)",
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint if available",
    )

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        sys.exit(1)

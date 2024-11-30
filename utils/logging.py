"""Logging configuration for the framework."""

import logging
from pathlib import Path
import sys


def setup_logging(level=logging.INFO):
    """Set up logging configuration."""
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Configure root logger first
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler with higher level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # Create file handler
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "experiment.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Force immediate output
    sys.stdout.flush()
    sys.stderr.flush()

    # Log a test message
    root_logger.info("Logging system initialized")


def get_logger(name):
    """Get logger for module."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # Ensure logger level is set
    return logger

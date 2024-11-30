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

    # Create console handler with higher level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Ensure stdout is flushed immediately
    console_handler.flush = sys.stdout.flush

    # Create file handler
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "experiment.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove any existing handlers
    root_logger.handlers = []

    # Add our handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Force immediate output
    sys.stdout.flush()

    # Log a test message
    root_logger.info("Logging system initialized")


def get_logger(name):
    """Get logger for module."""
    logger = logging.getLogger(name)
    # Ensure all messages are passed to handlers
    logger.propagate = True
    return logger

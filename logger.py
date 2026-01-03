"""
Logging configuration for Football Player Tracking Application.
Provides structured logging with console and optional file output.
"""

import logging
import sys
from config import LogConfig


def setup_logger(name: str = "futbl") -> logging.Logger:
    """
    Set up and return a configured logger instance.

    Args:
        name: Logger name (default: "futbl")

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Set log level from config
    level = getattr(logging, LogConfig.LEVEL.upper(), logging.INFO)
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        fmt=LogConfig.FORMAT,
        datefmt=LogConfig.DATE_FORMAT
    )

    # Console handler (always enabled)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if LogConfig.FILE:
        file_handler = logging.FileHandler(LogConfig.FILE)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Create default application logger
app_logger = setup_logger("futbl")


# Convenience functions for different log levels
def debug(msg: str, *args, **kwargs):
    """Log debug message."""
    app_logger.debug(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs):
    """Log info message."""
    app_logger.info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs):
    """Log warning message."""
    app_logger.warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs):
    """Log error message."""
    app_logger.error(msg, *args, **kwargs)


def critical(msg: str, *args, **kwargs):
    """Log critical message."""
    app_logger.critical(msg, *args, **kwargs)


# Module-specific loggers
def get_logger(module_name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        module_name: Name of the module (e.g., "futbl.camera", "futbl.ocr")

    Returns:
        Logger instance for the module
    """
    return setup_logger(f"futbl.{module_name}")

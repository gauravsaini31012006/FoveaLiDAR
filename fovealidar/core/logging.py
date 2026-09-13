"""
Logging setup for FoveaLiDAR.

Provides centralized logging configuration with file rotation,
console output, and structured formatting.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from fovealidar.core.config import get_config


def setup_logging(config_path: Optional[str] = None) -> logging.Logger:
    """
    Set up logging for the application.
    
    Args:
        config_path: Optional path to config file. If None, uses global config.
    
    Returns:
        Root logger configured for FoveaLiDAR.
    """
    config = get_config()
    log_config = config.logging
    
    # Create root logger
    root_logger = logging.getLogger("fovealidar")
    root_logger.setLevel(getattr(logging, log_config.level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt=log_config.format,
        datefmt=log_config.date_format
    )
    
    # Console handler
    if log_config.console_enabled:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_config.level.upper()))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_config.file_enabled:
        log_path = Path(log_config.file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=log_config.max_file_size_mb * 1024 * 1024,
            backupCount=log_config.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_config.level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    root_logger.propagate = False
    
    # Log startup message
    root_logger.info("=" * 60)
    root_logger.info("FoveaLiDAR logging initialized")
    root_logger.info(f"Log level: {log_config.level}")
    root_logger.info(f"Console logging: {log_config.console_enabled}")
    root_logger.info(f"File logging: {log_config.file_enabled}")
    if log_config.file_enabled:
        root_logger.info(f"Log file: {log_config.file_path}")
    root_logger.info("=" * 60)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        name: Module name (e.g., __name__)
    
    Returns:
        Logger instance.
    """
    return logging.getLogger(f"fovealidar.{name}")
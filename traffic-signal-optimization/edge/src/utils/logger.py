"""
Centralized logging configuration for edge device
Provides structured logging with rotation and multiple outputs
"""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger
from config.settings import EdgeSettings


class LoggerSetup:
    """Setup and configure logging for the application"""
    
    _initialized = False
    
    @classmethod
    def setup(cls, settings: Optional[EdgeSettings] = None) -> None:
        """
        Initialize logging configuration
        
        Args:
            settings: Application settings
        """
        if cls._initialized:
            return
        
        # Remove default logger
        logger.remove()
        
        # Get log level
        log_level = settings.log_level if settings else "INFO"
        
        # Console output with colors
        logger.add(
            sys.stderr,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
            level=log_level,
            colorize=True,
            backtrace=True,
            diagnose=True
        )
        
        # File output with rotation
        if settings and settings.log_file:
            log_file = Path(settings.log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            logger.add(
                str(log_file),
                format=(
                    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                    "{level: <8} | "
                    "{name}:{function}:{line} | "
                    "{message}"
                ),
                level=log_level,
                rotation=settings.log_rotation,
                retention=settings.log_retention,
                compression="zip",
                enqueue=True,  # Thread-safe
                backtrace=True,
                diagnose=True
            )
        
        cls._initialized = True
        logger.info("Logging system initialized")
    
    @staticmethod
    def get_logger(name: str):
        """Get a logger instance with the given name"""
        return logger.bind(name=name)


def get_logger(name: str = __name__):
    """
    Get a logger instance
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logger.bind(module=name)
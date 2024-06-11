"""
This module provides a function to set up logging for the application. It configures
both console and file logging handlers, allowing different log levels for each. The
log files are stored in a user-specific log directory.

"""
import logging
import os
from datetime import datetime

# Default log directory
DEFAULT_LOG_DIR = os.path.join(os.getcwd(), "logs")


def setup_logging(
    name: str,
    verbosity: int = logging.WARNING,
    user_log_path: str = None,
) -> logging.Logger:
    """
    Configures and returns a logger with specified log levels for console and file handlers.

    Args:
        name (str): The name of the logger.
        verbosity (int): Logging level (logging.DEBUG, logging.INFO, logging.WARNING, etc.).
                        Defaults to logging.INFO.
        user_log_path (str): Path to the user-specific log directory. Defaults to None.

    Returns:
        logging.Logger: The configured logger instance.

    The logger will always write DEBUG level messages to a file located in the user's log
    directory, while the console log level can be adjusted based on the 'verbosity' parameter.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # Set the minimum log level for the logger

    # Create console handler and set level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(verbosity)

    # Create formatter for console
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)

    # Add console handler to logger
    logger.addHandler(console_handler)

    # Determine log file path
    log_dir = user_log_path or DEFAULT_LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"sed_{datetime.now().strftime('%Y-%m-%d')}.log")

    # Create file handler and set level to debug
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create formatter for file
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # Add file handler to logger
    logger.addHandler(file_handler)

    # Capture warnings with the logging system
    logging.captureWarnings(True)

    return logger
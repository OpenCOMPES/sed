"""
This module provides a function to set up logging for the application. It configures
both console and file logging handlers, allowing different log levels for each. The
log files are stored in a user-specific log directory.

"""
import logging
import os
import sys
from datetime import datetime

# Default log directory
DEFAULT_LOG_DIR = os.path.join(os.getcwd(), "logs")
CONSOLE_VERBOSITY = logging.INFO
FILE_VERBOSITY = logging.DEBUG


def setup_logging(
    name: str,
    user_log_path: str = DEFAULT_LOG_DIR,
) -> logging.Logger:
    """
    Configures and returns a logger with specified log levels for console and file handlers.

    Args:
        name (str): The name of the logger.
        user_log_path (str): Path to the user-specific log directory. Defaults to DEFAULT_LOG_DIR.

    Returns:
        logging.Logger: The configured logger instance.

    The logger will always write DEBUG level messages to a file located in the user's log
    directory, while the console log level can be adjusted based on the 'verbosity' parameter.
    """
    # Create logger
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.INFO)  # Set the minimum log level for the logger

    # Create console handler and set level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(CONSOLE_VERBOSITY)

    # Create formatter for console
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    # Add console handler to logger
    logger.addHandler(console_handler)

    # Determine log file path
    os.makedirs(user_log_path, exist_ok=True)
    log_file = os.path.join(user_log_path, f"sed_{datetime.now().strftime('%Y-%m-%d')}.log")

    # Create file handler and set level to debug
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(FILE_VERBOSITY)

    # Create formatter for file
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s in %(filename)s:%(lineno)d",
    )
    file_handler.setFormatter(file_formatter)

    # Add file handler to logger
    logger.addHandler(file_handler)

    # Capture warnings with the logging system
    logging.captureWarnings(True)

    return logger

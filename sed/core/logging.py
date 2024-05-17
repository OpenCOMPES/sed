"""
This module provides a function to set up logging for the application. It configures
both console and file logging handlers, allowing different log levels for each. The
log files are stored in a user-specific log directory.

"""
import logging

from sed.core.user_dirs import USER_LOG_PATH


def setup_logging(name: str, verbose: bool = False, debug: bool = False) -> logging.Logger:
    """
    Configures and returns a logger with specified log levels for console and file handlers.

    Args:
        name (str): The name of the logger.
        verbose (bool): If True, sets the console log level to INFO. Defaults to False.
        debug (bool): If True, sets the console log level to DEBUG. Defaults to False.

    Returns:
        logging.Logger: The configured logger instance.

    The logger will always write DEBUG level messages to a file located in the user's log
    directory, while the console log level can be adjusted based on the 'verbose' and 'debug'
    flags.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set the minimum log level for the logger

    # Determine console log level
    if debug:
        console_level = logging.DEBUG
    elif verbose:
        console_level = logging.INFO
    else:
        console_level = logging.WARNING

    # Create console handler and set level to warning (default)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)

    # Create formatter for console
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)

    # Add console handler to logger
    logger.addHandler(console_handler)

    log_file = USER_LOG_PATH.joinpath("sed.log")

    # Create file handler and set level to debug
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create formatter for file
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # Add file handler to logger
    logger.addHandler(file_handler)

    return logger

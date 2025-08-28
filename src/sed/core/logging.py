"""
This module provides a function to set up logging for the application. It configures
both console and file logging handlers, allowing different log levels for each. The
log files are stored in a user-specific log directory.

"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from functools import wraps
from inspect import signature
from typing import Callable

# Default log directory
DEFAULT_LOG_DIR = os.path.join(os.getcwd(), "logs")
CONSOLE_VERBOSITY = logging.INFO
FILE_VERBOSITY = logging.DEBUG


def setup_logging(
    name: str,
    set_base_handler: bool = False,
    user_log_path: str | None = None,
) -> logging.Logger:
    """
    Configures and returns a logger with specified log levels for console and file handlers.

    Args:
        name (str): The name of the logger.
        set_base_handler (bool, optional): Option to re-initialize the base handler logging to the
            logfile. Defaults to False.
        user_log_path (str, optional): Path to the user-specific log directory.
            Defaults to DEFAULT_LOG_DIR.

    Returns:
        logging.Logger: The configured logger instance.

    The logger will always write DEBUG level messages to a file located in the user's log
    directory, while the console log level can be adjusted based on the 'verbosity' parameter.
    """
    # Create base logger
    base_logger = logging.getLogger("sed")
    base_logger.setLevel(logging.DEBUG)  # Set the minimum log level for the logger
    if set_base_handler or len(base_logger.handlers) == 0:
        if len(base_logger.handlers):
            base_logger.handlers.clear()

        # Determine log file path
        if user_log_path is None:
            user_log_path = DEFAULT_LOG_DIR
        try:
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
            base_logger.addHandler(file_handler)
        except (OSError, PermissionError):
            logging.warning(f"Cannot create logfile in Folder {user_log_path}, disabling logfile.")
            base_logger.addHandler(logging.NullHandler())
            base_logger.propagate = False

    # create named logger
    logger = base_logger.getChild(name)

    if logger.hasHandlers():
        logger.handlers.clear()

    # Create console handler and set level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(CONSOLE_VERBOSITY)

    # Create formatter for console
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    # Add console handler to logger
    logger.addHandler(console_handler)

    # Capture warnings with the logging system
    logging.captureWarnings(True)

    return logger


def set_verbosity(logger: logging.Logger, verbose: bool) -> None:
    """Sets log level for the given logger's default handler.

    Args:
        logger (logging.Logger): The logger on which to set the log level.
        verbose (bool): Sets loglevel to INFO if True, to WARNING otherwise.
    """
    handler = logger.handlers[0]
    if verbose:
        handler.setLevel(logging.INFO)
    else:
        handler.setLevel(logging.WARNING)


def call_logger(logger: logging.Logger):
    def log_call(func: Callable):
        @wraps(func)
        def new_func(*args, **kwargs):
            saved_args = locals()
            args_str = ""
            for arg in (
                saved_args["args"][1:]
                if "self" in signature(func).parameters
                else saved_args["args"]
            ):
                args_str += f"{arg}, "
            for name, arg in saved_args["kwargs"].items():
                args_str += f"{name}={arg}, "
            args_str = args_str.rstrip(", ")
            logger.debug(f"Call {func.__name__}({args_str})")
            return func(*args, **kwargs)

        return new_func

    return log_call

"""
This module sets up the user-specific directory paths for configuration, data, and logs
using the `platformdirs` library. The directories are created if they do not already exist.

Attributes:
    USER_CONFIG_PATH (pathlib.Path): The path to the user-specific configuration directory.
    USER_LOG_PATH (pathlib.Path): The path to the user-specific log directory.
    USER_DATA_PATH (pathlib.Path): The path to the user-specific data directory.
"""
from platformdirs import user_config_path
from platformdirs import user_data_path
from platformdirs import user_log_path

USER_CONFIG_PATH = user_config_path(appname="sed", appauthor="OpenCOMPES", ensure_exists=True)
USER_LOG_PATH = user_log_path(appname="sed", appauthor="OpenCOMPES", ensure_exists=True)
USER_DATA_PATH = user_data_path(appname="sed", appauthor="OpenCOMPES", ensure_exists=True)

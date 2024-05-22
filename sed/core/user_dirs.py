"""
This module sets up the user-specific directory paths for configuration, data, and logs
using the `platformdirs` library. The directories are created if they do not already exist.

Attributes:
    USER_CONFIG_PATH (pathlib.Path): The path to the user-specific configuration directory.
    USER_LOG_PATH (pathlib.Path): The path to the user-specific log directory.
    USER_DATA_PATH (pathlib.Path): The path to the user-specific data directory.
"""
from __future__ import annotations

from pathlib import Path

from platformdirs import user_config_path
from platformdirs import user_data_path
from platformdirs import user_log_path

USER_CONFIG_PATH = user_config_path(appname="sed", appauthor="OpenCOMPES", ensure_exists=True)
USER_LOG_PATH = user_log_path(appname="sed", appauthor="OpenCOMPES", ensure_exists=True)
USER_DATA_PATH = user_data_path(appname="sed", appauthor="OpenCOMPES", ensure_exists=True)


def construct_module_dirs(module: str, create_in: list = ["config", "data"]) -> dict[str, Path]:
    """
    Constructs module-specific subdirectories within specified user-specific directories.

    Args:
        module: Module subdirectory to append to the base paths.
        create_in (list): Specifies which base paths to use ('config', 'data', etc.).
        Defaults to ['config', 'data'].

    Returns:
        dict: A dictionary with keys as the path types ('config', 'data', etc.) and
        values as the constructed paths.
    """
    path_types = {"config": USER_CONFIG_PATH, "data": USER_DATA_PATH}
    constructed_paths = {}

    for path_type, base_path in path_types.items():
        if path_type in create_in:
            path = base_path.joinpath(module)
            path.mkdir(parents=True, exist_ok=True)
            constructed_paths[path_type] = path

    return constructed_paths

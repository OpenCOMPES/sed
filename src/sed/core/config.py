"""This module contains a config library for loading yaml/json files into dicts
"""
from __future__ import annotations

import copy
import json
import os
import platform
from importlib.util import find_spec
from pathlib import Path

import yaml
from platformdirs import user_config_path
from pydantic import ValidationError

from sed.core.config_model import ConfigModel
from sed.core.logging import setup_logging

package_dir = os.path.dirname(find_spec("sed").origin)

USER_CONFIG_PATH = user_config_path(appname="sed", appauthor="OpenCOMPES", ensure_exists=True)
SYSTEM_CONFIG_PATH = (
    Path(os.environ["ALLUSERSPROFILE"]).joinpath("sed")
    if platform.system() == "Windows"
    else Path("/etc/").joinpath("sed")
)
ENV_DIR = Path(".env")

# Configure logging
logger = setup_logging("config")


def parse_config(
    config: dict | str = None,
    folder_config: dict | str = None,
    user_config: dict | str = None,
    system_config: dict | str = None,
    default_config: (dict | str) = f"{package_dir}/config/default.yaml",
    verbose: bool = True,
    verify_config: bool = True,
) -> dict:
    """Load the config dictionary from a file, or pass the provided config dictionary.
    The content of the loaded config dictionary is then completed from a set of pre-configured
    config files in hierarchical order, by adding missing items. These additional config files
    are searched for in different places on the system as detailed below. Alternatively, they
    can be also passed as optional arguments (file path strings or dictionaries).

    Args:
        config (dict | str, optional): config dictionary or file path.
                Files can be *json* or *yaml*. Defaults to None.
        folder_config (dict | str, optional): working-folder-based config dictionary
            or file path. The loaded dictionary is completed with the folder-based values,
            taking preference over user, system and default values. Defaults to the file
            "sed_config.yaml" in the current working directory.
        user_config (dict | str, optional): user-based config dictionary
            or file path. The loaded dictionary is completed with the user-based values,
            taking preference over system and default values.
            Defaults to the file ".config/sed/config_v1.yaml" in the current user's home directory.
        system_config (dict | str, optional): system-wide config dictionary
            or file path. The loaded dictionary is completed with the system-wide values,
            taking preference over default values. Defaults to the file "/etc/sed/config_v1.yaml"
            on linux, and "%ALLUSERSPROFILE%/sed/config_v1.yaml" on windows.
        default_config (dict | str, optional): default config dictionary
            or file path. The loaded dictionary is completed with the default values.
            Defaults to *package_dir*/config/default.yaml".
        verbose (bool, optional): Option to report loaded config files. Defaults to True.
        verify_config (bool, optional): Option to verify config file. Defaults to True.
    Raises:
        TypeError: Raised if the provided file is neither *json* nor *yaml*.
        FileNotFoundError: Raised if the provided file is not found.
        ValueError: Raised if there is a validation error in the config file.

    Returns:
        dict: Loaded and completed config dict, possibly verified by pydantic config model.
    """
    if config is None:
        config = {}

    if isinstance(config, dict):
        config_dict = copy.deepcopy(config)
    else:
        config_dict = load_config(config)
        if verbose:
            logger.info(f"Configuration loaded from: [{str(Path(config).resolve())}]")

    folder_dict: dict = None
    if isinstance(folder_config, dict):
        folder_dict = copy.deepcopy(folder_config)
    else:
        if folder_config is None:
            folder_config = "./sed_config.yaml"
        if Path(folder_config).exists():
            folder_dict = load_config(folder_config)
            if verbose:
                logger.info(f"Folder config loaded from: [{str(Path(folder_config).resolve())}]")

    user_dict: dict = None
    if isinstance(user_config, dict):
        user_dict = copy.deepcopy(user_config)
    else:
        if user_config is None:
            user_config = str(USER_CONFIG_PATH.joinpath("config_v1.yaml"))
        if Path(user_config).exists():
            user_dict = load_config(user_config)
            if verbose:
                logger.info(f"User config loaded from: [{str(Path(user_config).resolve())}]")

    system_dict: dict = None
    if isinstance(system_config, dict):
        system_dict = copy.deepcopy(system_config)
    else:
        if system_config is None:
            system_config = str(SYSTEM_CONFIG_PATH.joinpath("config_v1.yaml"))
        if Path(system_config).exists():
            system_dict = load_config(system_config)
            if verbose:
                logger.info(f"System config loaded from: [{str(Path(system_config).resolve())}]")

    if isinstance(default_config, dict):
        default_dict = copy.deepcopy(default_config)
    else:
        default_dict = load_config(default_config)
        if verbose:
            logger.info(f"Default config loaded from: [{str(Path(default_config).resolve())}]")

    if folder_dict is not None:
        config_dict = complete_dictionary(
            dictionary=config_dict,
            base_dictionary=folder_dict,
        )
    if user_dict is not None:
        config_dict = complete_dictionary(
            dictionary=config_dict,
            base_dictionary=user_dict,
        )
    if system_dict is not None:
        config_dict = complete_dictionary(
            dictionary=config_dict,
            base_dictionary=system_dict,
        )
    config_dict = complete_dictionary(
        dictionary=config_dict,
        base_dictionary=default_dict,
    )

    if not verify_config:
        return config_dict

    try:
        # Run the config through the ConfigModel to ensure it is valid
        config_model = ConfigModel(**config_dict)
        return config_model.model_dump(exclude_unset=True, exclude_none=True)
    except ValidationError as e:
        error_msg = (
            "Invalid configuration file detected. The following validation errors were found:\n"
        )
        for error in e.errors():
            error_msg += f"\n- {' -> '.join(str(loc) for loc in error['loc'])}: {error['msg']}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def load_config(config_path: str) -> dict:
    """Loads config parameter files.

    Args:
        config_path (str): Path to the config file. Json or Yaml format are supported.

    Raises:
        FileNotFoundError: Raised if the config file cannot be found.
        TypeError: Raised if the provided file is neither *json* nor *yaml*.

    Returns:
        dict: loaded config dictionary
    """
    config_file = Path(config_path)
    if not config_file.is_file():
        error_message = f"could not find the configuration file: {config_file}"
        logger.error(error_message)
        raise FileNotFoundError(error_message)

    if config_file.suffix == ".json":
        with open(config_file, encoding="utf-8") as stream:
            config_dict = json.load(stream)
    elif config_file.suffix == ".yaml":
        with open(config_file, encoding="utf-8") as stream:
            config_dict = yaml.safe_load(stream)
    else:
        error_message = "config file must be of type json or yaml!"
        logger.error(error_message)
        raise TypeError(error_message)

    return config_dict


def save_config(config_dict: dict, config_path: str, overwrite: bool = False):
    """Function to save a given config dictionary to a json or yaml file. Normally, it loads any
    existing file of the given name, and keeps any existing dictionary keys not present in the
    provided dictionary. The overwrite option creates a fully empty dictionary first.

    Args:
        config_dict (dict): The dictionary to save.
        config_path (str): A string containing the path to the file where to save the dictionary
            to.
        overwrite (bool, optional): Option to overwrite an existing file with the given dictionary.
            Defaults to False.
    """
    config_file = Path(config_path)
    if config_file.is_file() and not overwrite:
        existing_config = load_config(config_path=config_path)
    else:
        existing_config = {}

    new_config = complete_dictionary(config_dict, existing_config)

    if config_file.suffix == ".json":
        with open(config_file, mode="w", encoding="utf-8") as stream:
            json.dump(new_config, stream, indent=2)
    elif config_file.suffix == ".yaml":
        with open(config_file, mode="w", encoding="utf-8") as stream:
            config_dict = yaml.dump(new_config, stream)
    else:
        raise TypeError("config file must be of type json or yaml!")


def complete_dictionary(dictionary: dict, base_dictionary: dict) -> dict:
    """Iteratively completes a dictionary from a base dictionary, by adding keys that are missing
    in the dictionary, and are present in the base dictionary.

    Args:
        dictionary (dict): the dictionary to be completed.
        base_dictionary (dict): the base dictionary.

    Returns:
        dict: the completed (merged) dictionary
    """
    if base_dictionary:
        for k, v in base_dictionary.items():
            if isinstance(v, dict):
                if k not in dictionary.keys():
                    dictionary[k] = v
                else:
                    if not isinstance(dictionary[k], dict):
                        raise ValueError(
                            "Cannot merge dictionaries. "
                            f"Mismatch on Key {k}: {dictionary[k]}, {v}.",
                        )
                    dictionary[k] = complete_dictionary(dictionary[k], v)
            else:
                if k not in dictionary.keys():
                    dictionary[k] = v

    return dictionary


def _parse_env_file(file_path: Path) -> dict:
    """Helper function to parse a .env file into a dictionary.

    Args:
        file_path (Path): Path to the .env file

    Returns:
        dict: Dictionary of environment variables from the file
    """
    env_content = {}
    if file_path.exists():
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if line and "=" in line:
                    key, val = line.split("=", 1)
                    env_content[key.strip()] = val.strip()
    return env_content


def read_env_var(var_name: str) -> str | None:
    """Read an environment variable from multiple locations in order:
    1. OS environment variables
    2. .env file in current directory
    3. .env file in user config directory
    4. .env file in system config directory

    Args:
        var_name (str): Name of the environment variable to read

    Returns:
        str | None: Value of the environment variable or None if not found
    """
    # 1. check OS environment variables
    value = os.getenv(var_name)
    if value is not None:
        logger.debug(f"Found {var_name} in OS environment variables")
        return value

    # 2. check .env in current directory
    local_vars = _parse_env_file(ENV_DIR)
    if var_name in local_vars:
        logger.debug(f"Found {var_name} in ./.env file")
        return local_vars[var_name]

    # 3. check .env in user config directory
    user_vars = _parse_env_file(USER_CONFIG_PATH / ".env")
    if var_name in user_vars:
        logger.debug(f"Found {var_name} in user config .env file")
        return user_vars[var_name]

    # 4. check .env in system config directory
    system_vars = _parse_env_file(SYSTEM_CONFIG_PATH / ".env")
    if var_name in system_vars:
        logger.debug(f"Found {var_name} in system config .env file")
        return system_vars[var_name]

    logger.debug(f"Environment variable {var_name} not found in any location")
    return None


def save_env_var(var_name: str, value: str) -> None:
    """Save an environment variable to the .env file in the user config directory.
    If the file exists, preserves other variables. If not, creates a new file.

    Args:
        var_name (str): Name of the environment variable to save
        value (str): Value to save for the environment variable
    """
    env_path = USER_CONFIG_PATH / ".env"
    env_content = _parse_env_file(env_path)

    # Update or add new variable
    env_content[var_name] = value

    # Write all variables back to file
    with open(env_path, "w") as f:
        for key, val in env_content.items():
            f.write(f"{key}={val}\n")
    logger.debug(f"Environment variable {var_name} saved to .env file")

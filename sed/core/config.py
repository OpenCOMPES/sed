"""This module contains a config library for loading yaml/json files into dicts
"""
import json
from pathlib import Path
from typing import Union

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "default.yaml"


def parse_config(config: Union[dict, str] = None) -> dict:
    """Load the default config dictionary, and update it if a user file or dict
    is provided.

    Args:
        config (Union[dict, str], optional): config dictionary or file path.
                Files can be *json* or *yaml*. Defaults to None.

    Returns:
        dict: Loaded and possibly completed config dictionary.
    """
    default_config = load_config(DEFAULT_CONFIG_PATH)

    if config is not None:
        user_config = (
            load_config(Path(config)) if isinstance(config, str) else config
        )
        insert_default_config(user_config, default_config)
    else:
        user_config = default_config

    return user_config


def load_config(config_path: Path) -> dict:
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
        raise FileNotFoundError(
            f"could not find the configuration file: {config_file}",
        )

    if config_file.suffix == ".json":
        with open(config_file, encoding="utf-8") as stream:
            config_dict = json.load(stream)
    elif config_file.suffix == ".yaml":
        with open(config_file, encoding="utf-8") as stream:
            config_dict = yaml.safe_load(stream)
    else:
        raise TypeError("config file must be of type json or yaml!")

    return config_dict


def insert_default_config(config: dict, default_config: dict) -> dict:
    """Inserts missing config parameters from a default config file.

    Args:
        config (dict): the config dictionary
        default_config (dict): the default config dictionary.

    Returns:
        dict: merged dictionary
    """
    for k, v in default_config.items():
        if isinstance(v, dict):
            if k not in config.keys():
                config[k] = v
            else:
                config[k] = insert_default_config(config[k], v)
        else:
            if k not in config.keys():
                config[k] = v

    return config

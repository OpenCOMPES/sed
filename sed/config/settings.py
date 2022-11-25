"""This module contains a settings/config library for loading yaml/json files into
dicts

"""
import json
import os
from importlib.util import find_spec
from pathlib import Path
from typing import Union

import yaml

package_dir = os.path.dirname(find_spec("sed").origin)


def parse_config(
    config: Union[dict, str] = None,
    default_config: Union[
        dict,
        str,
    ] = f"{package_dir}/config/default.yaml",
) -> dict:
    """Handle config dictionary or files.

    Args:
        config: config dictionary or file path.
                Files can be json or yaml.
        default_config: default config dictionary or file path.
                The loaded dictionary is completed with the default values.

    Raises:
        TypeError

    Returns:
        config_dict: loaded and possibly completed config dictionary.
    """

    if config is None:
        config = {}

    if isinstance(config, dict):
        config_dict = config
    else:
        config_dict = load_config(config)

    if isinstance(default_config, dict):
        default_dict = default_config
    else:
        default_dict = load_config(default_config)

    insert_default_config(config_dict, default_dict)

    return config_dict


def load_config(config_path: str) -> dict:
    """Loads config parameter files.

    Args:
        config_path: Path to the config file. Json or Yaml format are supported.

    Raises:
        TypeError, FileNotFoundError

    Returns:
        config_dict: loaded config dictionary
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
        config: the config dictionary
        default_config: the default config dictionary.

    Returns:
        config: merged dictionary
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

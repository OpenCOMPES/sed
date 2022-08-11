import json
import os
from pathlib import Path
from typing import Union

import yaml

import sed

package_dir = os.path.dirname(sed.__file__)


def parse_config(
    config: Union[dict, Path, str] = {},
    default_config: Union[
        dict,
        Path,
        str,
    ] = f"{package_dir}/config/default.yaml",
) -> dict:
    """Handle config dictionary or files.

    Args:
        config: config dictionary, file path or Path object.
                Files can be json or yaml.
        default_config: default config dictionary, file path Path object.
                The loaded dictionary is completed with the default values.

    Raises:
        TypeError

    Returns:
        config_dict: loaded and possibly completed config dictionary.
    """

    if isinstance(config, dict):
        config_dict = config
    else:
        if isinstance(config, str):
            config_file = Path(config)
        else:
            config_file = config

        if not isinstance(config_file, Path):
            raise TypeError(
                "config must be either a Path to a config file or a config dictionary!",
            )

        config_dict = load_config(config_file)

    if isinstance(default_config, dict):
        default_dict = default_config
    else:
        if isinstance(default_config, str):
            default_file = Path(default_config)
        else:
            default_file = default_config
        if not isinstance(default_file, Path):
            raise TypeError(
                "default_config must be either a Path to a config file or a config\
 dictionary!",
            )
        default_dict = load_config(default_file)

    insert_default_config(config_dict, default_dict)

    return config_dict


def load_config(config_file: Path) -> dict:
    """Loads config parameter files.

    Args:
        config_file: Path object to the config file. Json or Yaml format are supported.

    Raises:
        TypeError

    Returns:
        config_dict: loaded config dictionary
    """

    if not isinstance(config_file, Path):
        raise TypeError(
            "config_file must be a Path object!",
        )

    if config_file.suffix == ".json":
        with open(config_file) as stream:
            config_dict = json.load(stream)
    elif config_file.suffix == ".yaml":
        with open(config_file) as stream:
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

"""This is a code that performs several tests for the settings loader.

"""
import os
from importlib.util import find_spec

import pytest

from sed.core.config import load_config
from sed.core.config import parse_config

package_dir = os.path.dirname(find_spec("sed").origin)
DEFAULT_CONFIG_PATH = f"{package_dir}/core/default.yaml"

default_config_keys = [
    "binning",
    "histogram",
]
default_binning_keys = [
    "hist_mode",
    "mode",
    "pbar",
    "threads_per_worker",
    "threadpool_API",
]
default_histogram_keys = [
    "bins",
    "axes",
    "ranges",
]


def test_default_config():
    """Test the config loader for the default config."""
    config = parse_config()
    assert isinstance(config, dict)
    for key in default_config_keys:
        assert key in config.keys()
    for key in default_binning_keys:
        assert key in config["binning"].keys()
    for key in default_histogram_keys:
        assert key in config["histogram"].keys()


def test_load_dict():
    """Test the config loader for a dict."""
    config_dict = {"test_entry": True}
    config = parse_config(config_dict)
    assert isinstance(config, dict)
    for key in default_config_keys:
        assert key in config.keys()
    assert config["test_entry"] is True


def test_load_config():
    """Test if the config loader can handle json and yaml files."""
    config_json = load_config(
        f"{package_dir}/../tests/data/config/config.json",
    )
    config_yaml = load_config(
        f"{package_dir}/../tests/data/config/config.yaml",
    )
    assert config_json == config_yaml


def test_load_config_raise():
    """Test if the config loader raises an error for a wrong file type."""
    with pytest.raises(TypeError):
        load_config(f"{package_dir}/../README.md")


def test_insert_default_config():
    """Test the merging of a config and a default config dict"""
    default_config = load_config(DEFAULT_CONFIG_PATH)
    user_config = {
        "core": {"loader": "mpes"},
        "dataframe": None,
        "histogram": {"bins": 100},
    }
    updated_user_config = parse_config(config=user_config)
    assert isinstance(updated_user_config, dict)
    for key in ["core", "dataframe", "histogram"]:
        assert key in updated_user_config
    for key in default_config.keys():
        assert key in updated_user_config
    assert updated_user_config["core"] == {"loader": "mpes"}

"""This is a code that performs several tests for the settings loader.

"""
import os
from importlib.util import find_spec

import pytest

from sed.core.config import insert_default_config
from sed.core.config import load_config
from sed.core.config import parse_config

package_dir = os.path.dirname(find_spec("sed").origin)
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
    dict1 = {"key1": 1, "key2": 2, "nesteddict": {"key4": 4}}
    dict2 = {"key1": 2, "key3": 3, "nesteddict": {"key5": 5}}
    dict3 = insert_default_config(config=dict1, default_config=dict2)
    assert isinstance(dict3, dict)
    for key in ["key1", "key2", "key3", "nesteddict"]:
        assert key in dict3
    for key in ["key4", "key5"]:
        assert key in dict3["nesteddict"]
    assert dict3["key1"] == 1

"""Tests specific for Mpes loader"""
from __future__ import annotations

import logging
import os
from copy import deepcopy

import pytest

from sed.core.config import parse_config
from sed.loader.mpes.loader import MpesLoader

test_dir = os.path.join(os.path.dirname(__file__), "../..")
test_data_dir = os.path.join(test_dir, "data/loader/mpes")

config = parse_config(
    os.path.join(test_data_dir, "config.yaml"),
    folder_config={},
    user_config={},
    system_config={},
)


def test_channel_not_found_warning(caplog) -> None:
    """Test if the mpes loader gives the correct warning if a channel cannot be found."""
    ml = MpesLoader(config=config)

    with caplog.at_level(logging.WARNING):
        ml.read_dataframe(folders=test_data_dir)
    assert not caplog.messages

    # modify per_file channel
    config_ = deepcopy(config)
    config_["dataframe"]["channels"]["sampleBias"]["dataset_key"] = "invalid"
    ml = MpesLoader(config=config_)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        ml.read_dataframe(folders=test_data_dir)
    assert 'Entry "invalid" for channel "sampleBias" not found.' in caplog.messages[0]

    # modify per_electron channel
    config_ = deepcopy(config)
    config_["dataframe"]["channels"]["X"]["dataset_key"] = "invalid"
    ml = MpesLoader(config=config_)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        ml.read_dataframe(folders=test_data_dir)
    assert 'Entry "invalid" for channel "X" not found.' in caplog.messages[0]


def test_invalid_channel_format_raises() -> None:
    """Test if the mpes loader raises an exception if an illegal channel format is provided."""
    config_ = deepcopy(config)
    config_["dataframe"]["channels"]["sampleBias"]["format"] = "per_train"
    ml = MpesLoader(config=config_)

    with pytest.raises(ValueError) as e:
        ml.read_dataframe(folders=test_data_dir)

    expected_error = e.value.args[0]

    assert "Invalid 'format':per_train for channel sampleBias." in expected_error


def test_no_electron_channels_raises() -> None:
    """Test if the mpes loader raises an exception if no per-electron channels are provided."""
    config_ = deepcopy(config)
    config_["dataframe"]["channels"] = {
        "sampleBias": {"format": "per_file", "dataset_key": "KTOF:Lens:Sample:V"},
    }
    ml = MpesLoader(config=config_)

    with pytest.raises(ValueError) as e:
        ml.read_dataframe(folders=test_data_dir)

    expected_error = e.value.args[0]

    assert "No valid 'per_electron' channels found." in expected_error

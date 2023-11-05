import os
from importlib.util import find_spec
from pathlib import Path

import pytest

from sed.core.config import parse_config
from sed.loader.flash.loader import FlashLoader

package_dir = os.path.dirname(find_spec("sed").origin)
config_path = os.path.join(package_dir, "../tests/data/loader/flash/config.yaml")


@pytest.fixture
def config_file():
    return parse_config(config_path)


def test_get_channels_by_format(config_file):

    fl = FlashLoader(config_file)
    electron_channels = ["dldPosX", "dldPosY", "dldTimeSteps"]
    pulse_channels = [
        "sampleBias",
        "tofVoltage",
        "extractorVoltage",
        "extractorCurrent",
        "cryoTemperature",
        "sampleTemperature",
        "dldTimeBinSize",
    ]

    # Call get_channels_by_format method
    format_electron = fl.get_channels_by_format(["per_electron"])
    format_pulse = fl.get_channels_by_format(["per_pulse"])
    format_both = fl.get_channels_by_format(["per_pulse", "per_electron"])

    assert set(electron_channels) == set(format_electron)
    assert set(pulse_channels) == set(format_pulse)
    assert set(electron_channels + pulse_channels) == set(format_both)


@pytest.mark.parametrize(
    "sub_dir",
    ["online-0/fl1user3/", "express-0/fl1user3/", "FL1USER3/"],
)
def test_initialize_paths(config_file, fs, sub_dir):
    config = config_file
    del config["core"]["paths"]
    config["core"]["beamtime_id"] = "12345678"
    config["core"]["year"] = "2000"

    # find base path of beamline from config. here we use pg2
    base_path = config["dataframe"]["beamtime_dir"]["pg2"]
    expected_path = (
        Path(base_path) / config["core"]["year"] / "data" / config["core"]["beamtime_id"]
    )
    # create expected paths
    expected_raw_path = expected_path / "raw" / "hdf" / sub_dir
    expected_processed_path = expected_path / "processed" / "parquet"

    # create fake file system for testing
    fs.create_dir(expected_raw_path)
    fs.create_dir(expected_processed_path)

    # instance of class with correct config and call initialize_paths
    fl = FlashLoader(config=config)
    data_raw_dir, data_parquet_dir = fl.initialize_paths()
    print(data_raw_dir)
    assert expected_raw_path == data_raw_dir[0]
    assert expected_processed_path == data_parquet_dir

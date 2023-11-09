import os
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

import pytest

from sed.core.config import parse_config
from sed.loader.flash.loader import FlashLoader

package_dir = os.path.dirname(find_spec("sed").origin)
config_path = os.path.join(package_dir, "../tests/data/loader/flash/config.yaml")


@pytest.fixture
def config_file():
    return parse_config(config_path)


def test_get_channels_by_format(config_file: dict):

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
        "gmdTunnel",
    ]
    train_channels = ["timeStamp", "delayStage"]

    # Call get_channels_by_format method
    format_electron = fl.get_channels_by_format(["per_electron"])
    format_pulse = fl.get_channels_by_format(["per_pulse"])
    format_train = fl.get_channels_by_format(["per_train"])
    format_both = fl.get_channels_by_format(["per_pulse", "per_electron"])

    assert set(electron_channels) == set(format_electron)
    assert set(pulse_channels) == set(format_pulse)
    assert set(train_channels) == set(format_train)
    assert set(electron_channels + pulse_channels) == set(format_both)


@pytest.mark.parametrize(
    "sub_dir",
    ["online-0/fl1user3/", "express-0/fl1user3/", "FL1USER3/"],
)
def test_initialize_paths(
    config_file: dict,
    fs,
    sub_dir: Literal["online-0/fl1user3/", "express-0/fl1user3/", "FL1USER3/"],
):
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

    assert expected_raw_path == data_raw_dir[0]
    assert expected_processed_path == data_parquet_dir


def test_initialize_paths_filenotfound(config_file: dict):
    # test the FileNotFoundError
    config = config_file
    del config["core"]["paths"]
    config["core"]["beamtime_id"] = "11111111"
    config["core"]["year"] = "2000"

    # instance of class with correct config and call initialize_paths
    fl = FlashLoader(config=config)
    with pytest.raises(FileNotFoundError):
        _, _ = fl.initialize_paths()


def test_invalid_channel_format(config_file: dict):
    config = config_file
    config["dataframe"]["channels"]["dldPosX"]["format"] = "foo"

    fl = FlashLoader(config=config)

    with pytest.raises(ValueError):
        fl.read_dataframe()


def test_group_name_not_in_h5(config_file: dict):
    config = config_file
    config["dataframe"]["channels"]["dldPosX"]["group_name"] = "foo"
    h5_path = "FLASH1_USER3_stream_2_run43878_file1_20230130T153807.1.h5"
    fl = FlashLoader(config=config)

    with pytest.raises(ValueError) as e:
        fl.create_dataframe_per_file(config["core"]["paths"]["data_raw_dir"] + h5_path)

    assert str(e.value.args[0]) == "The group_name for channel dldPosX does not exist."


def test_buffer_schema_mismatch(config_file: dict):
    fl = FlashLoader(config=config_file)

    fl.read_dataframe(runs=["43878"])

    config = config_file
    config["dataframe"]["channels"]["gmdTunnel2"] = {
        "group_name": "/FL1/Photon Diagnostic/GMD/Pulse resolved energy/energy tunnel/",
        "format": "per_pulse",
    }

    with pytest.raises(ValueError) as e:
        fl.read_dataframe(runs=["43878"])
    expected_error = e.value.args
    assert "The available channels do not match the schema of file" in expected_error[0]
    assert expected_error[2] == "Missing in parquet: {'gmdTunnel2'}"
    assert expected_error[4] == "Please check the configuration file or set force_recreate to True."

    fl.read_dataframe(runs=["43878"], force_recreate=True)

"""Module tests.calibrator.delay, tests for the sed.calibrator.delay file
"""
import os
from importlib.util import find_spec

import pytest

from sed.calibrator.delay import DelayCalibrator
from sed.core.config import parse_config
from sed.loader.loader_interface import get_loader

package_dir = os.path.dirname(find_spec("sed").origin)
file = package_dir + "/../tests/data/loader/mpes/Scan0030_2.h5"
config = parse_config(
    package_dir + "/config/mpes_example_config.yaml",
    user_config={},
    system_config={},
)


def test_delay_parameters_from_file():
    """Test the option to extract the delay parameters from a file"""
    df, _ = get_loader(loader_name="mpes", config=config).read_dataframe(
        files=[file],
        collect_metadata=False,
    )
    dc = DelayCalibrator(config=config)
    df, metadata = dc.append_delay_axis(df, datafile=file)
    assert "delay" in df.columns
    assert "datafile" in metadata["calibration"]
    assert "delay_range" in metadata["calibration"]
    assert "adc_range" in metadata["calibration"]
    assert "time0" in metadata["calibration"]
    assert "delay_range_mm" in metadata["calibration"]


def test_delay_parameters_from_delay_range():
    """Test the option to extract the delay parameters from a delay range"""
    # from keywords
    df, _ = get_loader(loader_name="mpes", config=config).read_dataframe(
        files=[file],
        collect_metadata=False,
    )
    dc = DelayCalibrator(config=config)
    df, metadata = dc.append_delay_axis(df, delay_range=(-100, 200))
    assert "delay" in df.columns
    assert "delay_range" in metadata["calibration"]
    assert "adc_range" in metadata["calibration"]

    # from calibration
    df, _ = get_loader(loader_name="mpes", config=config).read_dataframe(
        files=[file],
        collect_metadata=False,
    )
    dc = DelayCalibrator(config=config)
    calibration = {"delay_range": (-100, 200), "adc_range": (100, 1000)}
    df, metadata = dc.append_delay_axis(df, calibration=calibration)
    assert "delay" in df.columns
    assert "delay_range" in metadata["calibration"]
    assert "adc_range" in metadata["calibration"]
    assert metadata["calibration"]["adc_range"] == (100, 1000)


def test_delay_parameters_from_delay_range_mm():
    """Test the option to extract the delay parameters from a mm range + t0"""
    # from keywords
    df, _ = get_loader(loader_name="mpes", config=config).read_dataframe(
        files=[file],
        collect_metadata=False,
    )
    dc = DelayCalibrator(config=config)
    with pytest.raises(NotImplementedError):
        dc.append_delay_axis(df, delay_range_mm=(1, 15))
    df, metadata = dc.append_delay_axis(df, delay_range_mm=(1, 15), time0=1)
    assert "delay" in df.columns
    assert "delay_range" in metadata["calibration"]
    assert "adc_range" in metadata["calibration"]
    assert "time0" in metadata["calibration"]
    assert "delay_range_mm" in metadata["calibration"]

    # from dict
    df, _ = get_loader(loader_name="mpes", config=config).read_dataframe(
        files=[file],
        collect_metadata=False,
    )
    dc = DelayCalibrator(config=config)
    calibration = {"delay_range_mm": (1, 15)}
    with pytest.raises(NotImplementedError):
        dc.append_delay_axis(df, calibration=calibration)
    calibration["time0"] = 1
    df, metadata = dc.append_delay_axis(df, calibration=calibration)
    assert "delay" in df.columns
    assert "delay_range" in metadata["calibration"]
    assert "adc_range" in metadata["calibration"]
    assert "time0" in metadata["calibration"]
    assert "delay_range_mm" in metadata["calibration"]

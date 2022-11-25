"""Module tests.calibrator.delay, tests for the sed.calibrator.delay file
"""
import os
from importlib.util import find_spec

import pytest

from sed.calibrator.delay import DelayCalibrator
from sed.config.settings import parse_config
from sed.loader.mpes import MpesLoader

package_dir = os.path.dirname(find_spec("sed").origin)
file = package_dir + "/../tests/data/loader/Scan0030_2.h5"
config = parse_config(package_dir + "/../tests/data/config/config.yaml")


def test_delay_parameters_from_file():
    """Test the option to extract the delay parameters from a file"""
    df = MpesLoader(config=config).read_dataframe(files=[file])
    dc = DelayCalibrator(config=config)
    dc.append_delay_axis(df, datafile=file)
    assert "delay" in df.columns


def test_delay_parameters_from_delay_range():
    """Test the option to extract the delay parameters from a delay range"""
    df = MpesLoader(config=config).read_dataframe(files=[file])
    dc = DelayCalibrator(config=config)
    dc.append_delay_axis(df, delay_range=(-100, 200))
    assert "delay" in df.columns


def test_delay_parameters_from_delay_range_mm():
    """Test the option to extract the delay parameters from a mm range + t0"""
    df = MpesLoader(config=config).read_dataframe(files=[file])
    dc = DelayCalibrator(config=config)
    with pytest.raises(NotImplementedError):
        dc.append_delay_axis(df, delay_range_mm=(1, 15))
    dc.append_delay_axis(df, delay_range_mm=(1, 15), time0=1)
    assert "delay" in df.columns

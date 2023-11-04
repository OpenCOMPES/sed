"""Module tests.calibrator.delay, tests for the sed.calibrator.delay file
"""
import os
from importlib.util import find_spec

import dask.dataframe
import numpy as np
import pandas as pd
import pytest

from sed.calibrator.delay import DelayCalibrator
from sed.core.config import parse_config
from sed.loader.loader_interface import get_loader

package_dir = os.path.dirname(find_spec("sed").origin)
file = package_dir + "/../tests/data/loader/mpes/Scan0030_2.h5"


def test_delay_parameters_from_file():
    """Test the option to extract the delay parameters from a file"""
    config = parse_config(
        config={
            "core": {"loader": "mpes"},
            "delay": {
                "p1_key": "@trARPES:DelayStage:p1",
                "p2_key": "@trARPES:DelayStage:p2",
                "t0_key": "@trARPES:DelayStage:t0",
            },
        },
        folder_config={},
        user_config={},
        system_config={},
    )
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
    config = parse_config(
        config={"core": {"loader": "mpes"}},
        folder_config={},
        user_config={},
        system_config={},
    )
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
    config = parse_config(
        config={"core": {"loader": "mpes"}},
        folder_config={},
        user_config={},
        system_config={},
    )
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


def test_loader_selection():
    """test that the correct calibration method is used based on the loader in the config"""
    config = parse_config(
        config={
            "core": {"loader": "mpes"},
            "delay": {
                "p1_key": "@trARPES:DelayStage:p1",
                "p2_key": "@trARPES:DelayStage:p2",
                "t0_key": "@trARPES:DelayStage:t0",
            },
        },
        folder_config={},
        user_config={},
        system_config={},
    )
    df, _ = get_loader(loader_name="mpes", config=config).read_dataframe(
        files=[file],
        collect_metadata=False,
    )
    dc = DelayCalibrator(config=config)
    assert dc.loader == "mpes"
    # assert dc.append_delay_axis.__doc__ == dc.append_delay_axis_mpes.__doc__
    assert getattr(dc, f"append_delay_axis_{dc.loader}") == dc.append_delay_axis_mpes

    config = parse_config(
        config={
            "core": {"loader": "hextof"},
            "delay": {"time0": 1},
        },
        folder_config={},
        user_config={},
        system_config={},
    )
    _ = dask.dataframe.from_pandas(
        pd.DataFrame({"delayStage": np.linspace(0, 1, 100)}),
        npartitions=2,
    )
    dc = DelayCalibrator(config=config)
    assert dc.loader == "hextof"
    # assert dc.append_delay_axis.__doc__ == dc.append_delay_axis_hextof.__doc__
    assert getattr(dc, f"append_delay_axis_{dc.loader}") == dc.append_delay_axis_hextof


def test_hextof_append_delay():
    """test functionality of the hextof delay calibration method"""
    config = parse_config(
        config={
            "core": {"loader": "hextof"},
            "dataframe": {"delay_column": "delay", "delay_stage_column": "delayStage"},
            "delay": {"time0": 1},
        },
        folder_config={},
        user_config={},
        system_config={},
    )
    df = dask.dataframe.from_pandas(
        pd.DataFrame({"dldPosX": np.linspace(0, 1, 100), "delayStage": np.linspace(0, 1, 100)}),
        npartitions=2,
    )
    dc = DelayCalibrator(config=config)
    df, metadata = dc.append_delay_axis(df)
    assert "delay" in df.columns
    assert "time0" in metadata["calibration"]
    assert metadata["calibration"]["time0"] == 1
    assert metadata["calibration"]["flip_time_axis"] is False
    np.testing.assert_allclose(df["delay"], np.linspace(0, 1, 100) - 1)


def test_correct_timing_fluctuation():
    """test that the timing fluctuation is corrected for correctly"""
    cfg = {
        "core": {"loader": "hextof"},
        "dataframe": {"delay_column": "delay", "delay_stage_column": "delayStage"},
        "delay": {
            "time0": 1,
            "fluctuations": {
                "bam": {
                    "sign": 1,
                    "preserve_mean": False,
                },
            },
        },
    }
    config = parse_config(
        config=cfg,
        folder_config={},
        user_config={},
        system_config={},
    )
    df = dask.dataframe.from_pandas(
        pd.DataFrame({"bam": np.random.normal(100) + 5, "delayStage": np.linspace(0, 1, 100)}),
        npartitions=2,
    )
    dc = DelayCalibrator(config=config)
    df, _ = dc.append_delay_axis(df)
    assert "delay" in df.columns
    df, meta = dc.correct_timing_fluctuation(df)
    expected = df["delayStage"] + df["bam"] - 1
    np.testing.assert_allclose(df["delay"], expected)

    cfg["delay"]["fluctuations"]["bam"]["preserve_mean"] = True
    config = parse_config(
        config=cfg,
        folder_config={},
        user_config={},
        system_config={},
    )
    dc = DelayCalibrator(config=config)
    df, _ = dc.append_delay_axis(df)
    assert "delay" in df.columns
    df, meta = dc.correct_timing_fluctuation(df)
    expected = df["delayStage"] - 1 + df["bam"] - 5

"""Module tests.calibrator.delay, tests for the sed.calibrator.delay file
"""
from __future__ import annotations

import os
from typing import Any

import dask.dataframe
import numpy as np
import pandas as pd
import pytest

from sed.calibrator.delay import DelayCalibrator
from sed.core.config import parse_config
from sed.loader.loader_interface import get_loader

test_dir = os.path.join(os.path.dirname(__file__), "..")
file = test_dir + "/data/loader/mpes/Scan0030_2.h5"


def test_delay_parameters_from_file() -> None:
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
    df, _, _ = get_loader(loader_name="mpes", config=config).read_dataframe(
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


def test_delay_parameters_from_delay_range() -> None:
    """Test the option to extract the delay parameters from a delay range"""
    # from keywords
    config = parse_config(
        config={"core": {"loader": "mpes"}},
        folder_config={},
        user_config={},
        system_config={},
    )
    df, _, _ = get_loader(loader_name="mpes", config=config).read_dataframe(
        files=[file],
        collect_metadata=False,
    )
    dc = DelayCalibrator(config=config)
    df, metadata = dc.append_delay_axis(df, delay_range=(-100, 200))
    assert "delay" in df.columns
    assert "delay_range" in metadata["calibration"]
    assert "adc_range" in metadata["calibration"]

    # from calibration
    df, _, _ = get_loader(loader_name="mpes", config=config).read_dataframe(
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


def test_delay_parameters_from_delay_range_mm() -> None:
    """Test the option to extract the delay parameters from a mm range + t0"""
    # from keywords
    config = parse_config(
        config={"core": {"loader": "mpes"}},
        folder_config={},
        user_config={},
        system_config={},
    )
    df, _, _ = get_loader(loader_name="mpes", config=config).read_dataframe(
        files=[file],
        collect_metadata=False,
    )
    dc = DelayCalibrator(config=config)
    with pytest.raises(NotImplementedError):
        dc.append_delay_axis(df, delay_range_mm=(1, 15))
    df, metadata = dc.append_delay_axis(
        df,
        delay_range_mm=(1, 15),
        time0=1,
        adc_range=config["delay"]["adc_range"],
    )
    assert "delay" in df.columns
    assert "delay_range" in metadata["calibration"]
    assert "adc_range" in metadata["calibration"]
    assert "time0" in metadata["calibration"]
    assert "delay_range_mm" in metadata["calibration"]

    # from dict
    df, _, _ = get_loader(loader_name="mpes", config=config).read_dataframe(
        files=[file],
        collect_metadata=False,
    )
    dc = DelayCalibrator(config=config)
    calibration: dict[str, Any] = {"delay_range_mm": (1, 15)}
    with pytest.raises(NotImplementedError):
        dc.append_delay_axis(df, calibration=calibration)
    calibration["time0"] = 1
    df, metadata = dc.append_delay_axis(df, calibration=calibration)
    assert "delay" in df.columns
    assert "delay_range" in metadata["calibration"]
    assert "adc_range" in metadata["calibration"]
    assert "time0" in metadata["calibration"]
    assert "delay_range_mm" in metadata["calibration"]


bam_vals = 1000 * (np.random.normal(size=100) + 5)
delay_stage_vals = np.linspace(0, 99, 100)
cfg = {
    "core": {"loader": "flash"},
    "dataframe": {"columns": {"delay": "delay"}},
    "delay": {
        "offsets": {
            "constant": 1,
            "flip_delay_axis": True,
            "columns": {
                "bam": {"weight": 0.001, "preserve_mean": False},
            },
        },
    },
}
test_dataframe = dask.dataframe.from_pandas(
    pd.DataFrame(
        {
            "bam": bam_vals.copy(),
            "delay": delay_stage_vals.copy(),
        },
    ),
    npartitions=2,
)


def test_add_offset_from_config(df=test_dataframe) -> None:
    """test that the timing offset is corrected for correctly from config"""
    config = parse_config(
        config=cfg,
        folder_config={},
        user_config={},
        system_config={},
    )

    expected = -np.asarray(delay_stage_vals + bam_vals * 0.001 + 1)

    dc = DelayCalibrator(config=config)
    df, _ = dc.add_offsets(df.copy())
    assert "delay" in df.columns
    assert "bam" in dc.offsets["columns"].keys()
    np.testing.assert_allclose(expected, df["delay"])


def test_add_offset_from_args(df=test_dataframe) -> None:
    """test that the timing offset applied with arguments works"""
    cfg_ = cfg.copy()
    cfg_.pop("delay")
    config = parse_config(
        config=cfg_,
        folder_config={},
        user_config={},
        system_config={},
    )
    dc = DelayCalibrator(config=config)
    df, _ = dc.add_offsets(
        df.copy(),
        constant=1,
        flip_delay_axis=True,
        columns="bam",
    )
    assert "delay" in df.columns
    assert "bam" in dc.offsets["columns"].keys()
    expected = -np.array(
        delay_stage_vals + bam_vals * 1 + 1,
    )
    np.testing.assert_allclose(expected, df["delay"])


def test_add_offset_from_dict(df=test_dataframe) -> None:
    """test that the timing offset is corrected for correctly from config"""
    cfg_ = cfg.copy()
    offsets = cfg["delay"]["offsets"]  # type:ignore
    offsets["columns"]["bam"].pop("weight")
    offsets["flip_delay_axis"] = False
    cfg_.pop("delay")
    config = parse_config(
        config=cfg_,
        folder_config={},
        user_config={},
        system_config={},
    )

    expected = np.asarray(delay_stage_vals + bam_vals * 1 + 1)

    dc = DelayCalibrator(config=config)
    df, _ = dc.add_offsets(df.copy(), offsets=offsets)
    assert "delay" in df.columns
    assert "bam" in dc.offsets["columns"].keys()
    np.testing.assert_allclose(expected, df["delay"])

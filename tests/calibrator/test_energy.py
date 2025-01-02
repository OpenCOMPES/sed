"""Module tests.calibrator.energy, tests for the sed.calibrator.energy module
"""
from __future__ import annotations

import csv
import glob
import itertools
import os
from copy import deepcopy
from typing import Any
from typing import Literal

import dask.dataframe
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sed.calibrator.energy import EnergyCalibrator
from sed.core.config import parse_config
from sed.loader.loader_interface import get_loader

test_dir = os.path.join(os.path.dirname(__file__), "..")
df_folder = test_dir + "/data/loader/mpes/"
folder = test_dir + "/data/calibrator/"
files = glob.glob(df_folder + "*.h5")

traces_list = []
with open(folder + "traces.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        traces_list.append(row)
traces = np.asarray(traces_list).T
with open(folder + "tof.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    tof = np.asarray(next(reader))
with open(folder + "biases.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    biases = np.asarray(next(reader))


def test_bin_data_and_read_biases_from_files() -> None:
    """Test binning the data and extracting the bias values from the files"""
    config = parse_config(
        config={"dataframe": {"tof_binning": 2}, "energy": {"bias_key": "@KTOF:Lens:Sample:V"}},
        folder_config={},
        user_config={},
        system_config={},
    )
    ec = EnergyCalibrator(
        config=config,
        loader=get_loader("mpes", config=config),
    )
    ec.bin_data(data_files=files)
    assert ec.traces.ndim == 2
    assert ec.traces.shape == (2, 1000)
    assert ec.tof.ndim == 1
    assert ec.biases.ndim == 1
    assert len(ec.biases) == 2
    default_config = parse_config(config={}, folder_config={}, user_config={}, system_config={})
    ec = EnergyCalibrator(
        config=default_config,
        loader=get_loader("mpes", config=default_config),
    )
    with pytest.raises(ValueError):
        ec.bin_data(data_files=files)
    faulty_config = parse_config(
        config={"energy": {"bias_key": "@KTOF:Lens:Sample"}},
        folder_config={},
        user_config={},
        system_config={},
    )
    ec = EnergyCalibrator(
        config=faulty_config,
        loader=get_loader("mpes", config=faulty_config),
    )
    with pytest.raises(ValueError):
        ec.bin_data(data_files=files)


def test_energy_calibrator_from_arrays_norm() -> None:
    """Test loading the energy, bias and tof traces into the class"""
    config = parse_config(
        config={"dataframe": {"tof_binning": 2}},
        folder_config={},
        user_config={},
        system_config={},
    )
    ec = EnergyCalibrator(
        config=config,
        loader=get_loader("mpes", config=config),
    )
    ec.load_data(biases=biases, traces=traces, tof=tof)
    assert len(ec.biases) == 11
    assert ec.traces.shape == (11, 1000)
    ec.normalize()
    for i in range(ec.traces_normed.shape[0]):
        assert np.max(ec.traces_normed[i, :]) == 1


def test_feature_extract() -> None:
    """Test generating the ranges, and extracting the features"""
    config = parse_config(
        config={"dataframe": {"tof_binning": 2}},
        folder_config={},
        user_config={},
        system_config={},
    )
    rand = [int((i + 1) * (20 + np.random.random() * 10)) for i in range(11)]
    traces_rand = np.zeros((len(rand), traces.shape[1]))
    for i, rnd in enumerate(rand):
        traces_rand[i, rnd:] = traces[0, 0:-rnd]
    ref_id = np.random.randint(0, 10)
    rng = (
        64500 + (tof[1] - tof[0]) * rand[ref_id],
        65300 + (tof[1] - tof[0]) * rand[ref_id],
    )
    ec = EnergyCalibrator(
        config=config,
        loader=get_loader("mpes", config=config),
    )
    ec.load_data(biases=biases, traces=traces_rand, tof=tof)
    ec.add_ranges(ranges=rng, ref_id=ref_id)
    for pos, feat_rng in zip(rand, ec.featranges):
        assert feat_rng[0] < (tof[1] - tof[0]) * pos + 65000 < feat_rng[1]
    ec.feature_extract()
    diff = ec.peaks[0, 0] - ((tof[1] - tof[0]) * rand[0] + 65000)
    np.testing.assert_allclose(
        ec.peaks[:, 0],
        ((tof[1] - tof[0]) * np.asarray(rand) + 65000) + diff,
    )

    # illegal keywords
    with pytest.raises(TypeError):
        ec.add_ranges(ranges=rng, ref_id=ref_id, illegal_kwd=True)


def test_adjust_ranges() -> None:
    """Test the interactive function for adjusting the feature ranges"""
    config = parse_config(
        config={"dataframe": {"tof_binning": 2}},
        folder_config={},
        user_config={},
        system_config={},
    )
    rand = [int((i + 1) * (20 + np.random.random() * 10)) for i in range(11)]
    traces_rand = np.zeros((len(rand), traces.shape[1]))
    for i, rnd in enumerate(rand):
        traces_rand[i, rnd:] = traces[0, 0:-rnd]
    ref_id = np.random.randint(0, 10)
    rng = (
        64500 + (tof[1] - tof[0]) * rand[ref_id],
        65300 + (tof[1] - tof[0]) * rand[ref_id],
    )
    ec = EnergyCalibrator(
        config=config,
        loader=get_loader("mpes", config=config),
    )
    ec.load_data(biases=biases, traces=traces_rand, tof=tof)
    ec.adjust_ranges(ranges=rng, ref_id=ref_id, apply=True)
    for pos, feat_rng in zip(rand, ec.featranges):
        assert feat_rng[0] < (tof[1] - tof[0]) * pos + 65000 < feat_rng[1]
    diff = ec.peaks[0, 0] - ((tof[1] - tof[0]) * rand[0] + 65000)
    np.testing.assert_allclose(
        ec.peaks[:, 0],
        ((tof[1] - tof[0]) * np.asarray(rand) + 65000) + diff,
    )

    # illegal keywords
    with pytest.raises(TypeError):
        ec.adjust_ranges(ranges=rng, ref_id=ref_id, apply=True, illegal_kwd=True)


energy_scales = ["kinetic", "binding"]
calibration_methods = ["lmfit", "lstsq", "lsqr"]


@pytest.mark.parametrize(
    "energy_scale, calibration_method",
    itertools.product(energy_scales, calibration_methods),
)
def test_calibrate_append(energy_scale: str, calibration_method: str) -> None:
    """Test if the calibration routines generate the correct slope of energy vs. tof,
    and the application to the data frame.

    Args:
        energy_scale (str): type of energy scaling
        calibration_method (str): method used for calibration
    """
    config = parse_config(
        config={"dataframe": {"tof_binning": 4}},
        folder_config={},
        user_config={},
        system_config={},
    )
    loader = get_loader(loader_name="mpes", config=config)
    df, _, _ = loader.read_dataframe(folders=df_folder, collect_metadata=False)
    ec = EnergyCalibrator(config=config, loader=loader)
    ec.load_data(biases=biases, traces=traces, tof=tof)
    ec.normalize()
    rng = (66100, 67000)
    ref_id = 5
    ec.add_ranges(ranges=rng, ref_id=ref_id)
    ec.feature_extract()
    e_ref = -0.5
    calibdict = ec.calibrate(
        ref_energy=e_ref,
        energy_scale=energy_scale,
        method=calibration_method,
    )
    df, metadata = ec.append_energy_axis(df)
    assert config["dataframe"]["columns"]["energy"] in df.columns
    axis = calibdict["axis"]
    diff = np.diff(axis)
    if energy_scale == "kinetic":
        assert np.all(diff < 0)
    else:
        assert np.all(diff > 0)

    for key, value in ec.calibration.items():
        np.testing.assert_equal(
            metadata["calibration"][key],
            value,
        )

    # illegal keywords
    with pytest.raises(TypeError):
        calibdict = ec.calibrate(
            ref_energy=e_ref,
            energy_scale=energy_scale,
            method=calibration_method,
            illegal_kwd=True,
        )


calib_types = ["fit", "poly"]
calib_dicts = [{"d": 1, "t0": 0, "E0": 0}, {"coeffs": [1, 2, 3], "E0": 0}]


@pytest.mark.parametrize(
    "calib_type, calib_dict",
    zip(calib_types, calib_dicts),
)
def test_append_energy_axis_from_dict_kwds(calib_type: str, calib_dict: dict) -> None:
    """Function to test if the energy calibration is correctly applied using a dict or
    kwd parameters.

    Args:
        calib_type (str): type of calibration.
        calib_dict (dict): Dictionary with calibration parameters.
    """
    config = parse_config(config={}, folder_config={}, user_config={}, system_config={})
    loader = get_loader(loader_name="mpes", config=config)
    # from dict
    df, _, _ = loader.read_dataframe(folders=df_folder, collect_metadata=False)
    ec = EnergyCalibrator(config=config, loader=loader)
    df, metadata = ec.append_energy_axis(df, calibration=calib_dict)
    assert config["dataframe"]["columns"]["energy"] in df.columns

    for key in calib_dict:
        np.testing.assert_equal(metadata["calibration"][key], calib_dict[key])
    assert metadata["calibration"]["calib_type"] == calib_type

    # from kwds
    df, _, _ = loader.read_dataframe(folders=df_folder, collect_metadata=False)
    ec = EnergyCalibrator(config=config, loader=loader)
    df, metadata = ec.append_energy_axis(df, **calib_dict)
    assert config["dataframe"]["columns"]["energy"] in df.columns

    for key in calib_dict:
        np.testing.assert_equal(metadata["calibration"][key], calib_dict[key])
    assert metadata["calibration"]["calib_type"] == calib_type


def test_append_energy_axis_raises() -> None:
    """Test if apply_correction raises the correct errors"""
    config = parse_config(config={}, folder_config={}, user_config={}, system_config={})
    loader = get_loader(loader_name="mpes", config=config)
    df, _, _ = loader.read_dataframe(folders=df_folder, collect_metadata=False)
    ec = EnergyCalibrator(config=config, loader=loader)
    with pytest.raises(ValueError):
        df, _ = ec.append_energy_axis(df, calibration={"d": 1, "t0": 0})
    with pytest.raises(NotImplementedError):
        df, _ = ec.append_energy_axis(
            df,
            calibration={"d": 1, "t0": 0, "E0": 0, "calib_type": "invalid"},
        )


def test_append_tof_ns_axis() -> None:
    """Function to test if the tof_ns calibration is correctly applied.
    TODO: add further tests once the discussion about units is done.
    """
    cfg = {
        "dataframe": {
            "columns": {
                "tof": "t",
                "tof_ns": "t_ns",
            },
            "tof_binning": 2,
            "tof_binwidth": 1e-9,
        },
    }
    config = parse_config(config=cfg, folder_config={}, user_config={}, system_config={})
    loader = get_loader(loader_name="mpes", config=config)

    # from kwds
    df, _, _ = loader.read_dataframe(folders=df_folder, collect_metadata=False)
    ec = EnergyCalibrator(config=config, loader=loader)
    df, _ = ec.append_tof_ns_axis(df, binwidth=2e-9, binning=2)
    assert config["dataframe"]["columns"]["tof_ns"] in df.columns
    np.testing.assert_allclose(df[ec.tof_column], df[ec.tof_ns_column] / 4)

    # from config
    df, _, _ = loader.read_dataframe(folders=df_folder, collect_metadata=False)
    ec = EnergyCalibrator(config=config, loader=loader)
    df, _ = ec.append_tof_ns_axis(df)
    assert config["dataframe"]["columns"]["tof_ns"] in df.columns
    np.testing.assert_allclose(df[ec.tof_column], df[ec.tof_ns_column] / 2)

    # illegal keywords:
    df, _, _ = loader.read_dataframe(folders=df_folder, collect_metadata=False)
    ec = EnergyCalibrator(config=config, loader=loader)
    with pytest.raises(TypeError):
        df, _ = ec.append_tof_ns_axis(df, illegal_kwd=True)


amplitude = 2.5
center = (730, 730)
sample = np.array(
    [
        [0, center[0], 2 * center[0], center[0], center[0], center[0]],
        [center[1], center[1], center[1], 0, center[1], 2 * center[1]],
        [0, 0, 0, 0, 0, 0],
    ],
).T
columns = ["X", "Y", "t"]
dims = ["X", "Y", "t"]
shape = (100, 100, 100)
tof_fermi = 132250  # pylint: disable=invalid-name
coords = [
    np.linspace(0, 2048, 100),
    np.linspace(0, 2048, 100),
    np.linspace(tof_fermi - 1000, tof_fermi + 1000, 100),
]
image = xr.DataArray(
    data=np.random.rand(*(100, 100, 100)),
    coords=dict(zip(dims, coords)),
)

correction_types = [
    "spherical",
    "Lorentzian",
    "Gaussian",
    "Lorentzian_asymmetric",
]
correction_kwds = [
    {"diameter": 5000},
    {"gamma": 900},
    {"sigma": 500},
    {"gamma": 900, "gamma2": 900, "amplitude2": amplitude},
]


@pytest.mark.parametrize(
    "correction_type, correction_kwd",
    zip(correction_types, correction_kwds),
)
def test_energy_correction(correction_type: str, correction_kwd: dict) -> None:
    """Function to test if all energy correction functions generate symmetric curves
    with the maximum at the center x/y.

    Args:
        correction_type (str): type of correction to test
        correction_kwd (dict): parameters to pass to the function
    """
    # From keywords
    config = parse_config(config={}, user_config={}, system_config={})
    sample_df = pd.DataFrame(sample, columns=columns)
    ec = EnergyCalibrator(
        config=config,
        loader=get_loader("mpes", config=config),
    )
    ec.adjust_energy_correction(
        image=image,
        correction_type=correction_type,
        center=center,
        amplitude=amplitude,
        tof_fermi=tof_fermi,
        apply=True,
        **correction_kwd,
    )
    df, metadata = ec.apply_energy_correction(sample_df)
    t = df[config["dataframe"]["columns"]["corrected_tof"]]
    assert t[0] == t[2]
    assert t[0] < t[1]
    assert t[3] == t[5]
    assert t[3] < t[4]
    assert t[1] == t[4]

    assert ec.correction["correction_type"] == correction_type
    assert ec.correction["amplitude"] == amplitude
    assert ec.correction["center"] == center

    for key, value in ec.correction.items():
        np.testing.assert_equal(
            metadata["correction"][key],
            value,
        )
    # From dict
    config = parse_config(config={}, user_config={}, system_config={})
    sample_df = pd.DataFrame(sample, columns=columns)
    ec = EnergyCalibrator(
        config=config,
        loader=get_loader("mpes", config=config),
    )
    correction: dict[Any, Any] = {
        "correction_type": correction_type,
        "amplitude": amplitude,
        "center": center,
        **correction_kwd,
    }
    ec.adjust_energy_correction(
        image=image,
        tof_fermi=tof_fermi,
        apply=True,
        **correction,
    )
    df, metadata = ec.apply_energy_correction(sample_df)
    t = df[config["dataframe"]["columns"]["corrected_tof"]]
    assert t[0] == t[2]
    assert t[0] < t[1]
    assert t[3] == t[5]
    assert t[3] < t[4]
    assert t[1] == t[4]

    assert ec.correction["correction_type"] == correction["correction_type"]
    assert ec.correction["amplitude"] == correction["amplitude"]
    assert ec.correction["center"] == correction["center"]

    for key, value in ec.correction.items():
        np.testing.assert_equal(
            metadata["correction"][key],
            value,
        )


@pytest.mark.parametrize(
    "correction_type",
    correction_types,
)
def test_adjust_energy_correction_raises(correction_type: str) -> None:
    """Function to test if the adjust_energy_correction function raises the correct errors.

    Args:
        correction_type (str): type of correction to test
    """
    config = parse_config(config={}, folder_config={}, user_config={}, system_config={})
    ec = EnergyCalibrator(
        config=config,
        loader=get_loader("mpes", config=config),
    )
    correction_dict: dict[str, Any] = {
        "correction_type": correction_type,
        "amplitude": amplitude,
        "center": center,
    }
    with pytest.raises(ValueError):
        ec.adjust_energy_correction(
            image=image,
            **correction_dict,
            apply=True,
        )
    if correction_type == "Lorentzian_asymmetric":
        correction_dict = {
            "correction_type": correction_type,
            "amplitude": amplitude,
            "center": center,
            "gamma": 900,
        }
        ec.adjust_energy_correction(
            image=image,
            **correction_dict,
            apply=True,
        )
        assert ec.correction["gamma2"] == correction_dict["gamma"]
        assert ec.correction["amplitude2"] == correction_dict["amplitude"]


@pytest.mark.parametrize(
    "correction_type, correction_kwd",
    zip(correction_types, correction_kwds),
)
def test_energy_correction_from_dict_kwds(correction_type: str, correction_kwd: dict) -> None:
    """Function to test if the energy correction is correctly applied using a dict or
    kwd parameters.

    Args:
        correction_type (str): type of correction to test
        correction_kwd (dict): parameters to pass to the function
    """
    config = parse_config(config={}, folder_config={}, user_config={}, system_config={})
    sample_df = pd.DataFrame(sample, columns=columns)
    ec = EnergyCalibrator(
        config=config,
        loader=get_loader("mpes", config=config),
    )
    correction_dict: dict[str, Any] = {
        "correction_type": correction_type,
        "amplitude": amplitude,
        "center": center,
        **correction_kwd,
    }
    df, metadata = ec.apply_energy_correction(
        sample_df,
        correction=correction_dict,
    )
    t = df[config["dataframe"]["columns"]["corrected_tof"]]
    assert t[0] == t[2]
    assert t[0] < t[1]
    assert t[3] == t[5]
    assert t[3] < t[4]
    assert t[1] == t[4]

    for key, value in correction_dict.items():
        np.testing.assert_equal(
            metadata["correction"][key],
            value,
        )

    # from kwds
    sample_df = pd.DataFrame(sample, columns=columns)
    ec = EnergyCalibrator(
        config=config,
        loader=get_loader("mpes", config=config),
    )
    df, metadata = ec.apply_energy_correction(sample_df, **correction_dict)
    t = df[config["dataframe"]["columns"]["corrected_tof"]]
    assert t[0] == t[2]
    assert t[0] < t[1]
    assert t[3] == t[5]
    assert t[3] < t[4]
    assert t[1] == t[4]

    for key, value in correction_dict.items():
        np.testing.assert_equal(
            metadata["correction"][key],
            value,
        )


@pytest.mark.parametrize(
    "correction_type",
    correction_types,
)
def test_apply_energy_correction_raises(correction_type: str) -> None:
    """Function to test if the apply_energy_correction raises the correct errors.

    Args:
        correction_type (str): type of correction to test
    """
    config = parse_config(config={}, folder_config={}, user_config={}, system_config={})
    sample_df = pd.DataFrame(sample, columns=columns)
    ec = EnergyCalibrator(
        config=config,
        loader=get_loader("mpes", config=config),
    )
    correction_dict: dict[str, Any] = {
        "correction_type": correction_type,
        "amplitude": amplitude,
        "center": center,
    }
    with pytest.raises(ValueError):
        df, _ = ec.apply_energy_correction(
            sample_df,
            correction=correction_dict,
        )
    if correction_type == "Lorentzian_asymmetric":
        correction_dict = {
            "correction_type": correction_type,
            "amplitude": amplitude,
            "center": center,
            "gamma": 900,
        }
        df, _ = ec.apply_energy_correction(
            sample_df,
            correction=correction_dict,
        )
        assert config["dataframe"]["columns"]["corrected_tof"] in df.columns


@pytest.mark.parametrize(
    "energy_scale",
    ["kinetic", "binding"],
)
def test_add_offsets_functionality(energy_scale: str) -> None:
    """test the add_offsets function"""
    scale_sign: Literal[-1, 1] = -1 if energy_scale == "binding" else 1
    config = parse_config(
        config={
            "energy": {
                "calibration": {
                    "energy_scale": energy_scale,
                },
                "offsets": {
                    "constant": 1,
                    "columns": {
                        "off1": {
                            "weight": 1,
                            "preserve_mean": True,
                        },
                        "off2": {"weight": -1, "preserve_mean": False},
                        "off3": {"weight": 1, "preserve_mean": False, "reduction": "mean"},
                    },
                },
            },
        },
        folder_config={},
        user_config={},
        system_config={},
    )
    params = {
        "constant": 1,
        "energy_column": "energy",
        "columns": ["off1", "off2", "off3"],
        "weights": [1, -1, 1],
        "preserve_mean": [True, False, False],
        "reductions": [None, None, "mean"],
    }
    df = pd.DataFrame(
        {
            "energy": [10, 20, 30, 40, 50, 60],
            "off1": [1, 2, 3, 4, 5, 6],
            "off2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "off3": [10.1, 10.2, 10.3, 10.4, 10.5, 10.6],
        },
    )
    t_df = dask.dataframe.from_pandas(df.copy(), npartitions=2)
    ec = EnergyCalibrator(
        config=config,
        loader=get_loader("flash", config=config),
    )
    res, meta = ec.add_offsets(t_df)
    exp_vals = df["energy"].copy() + 1 * scale_sign
    exp_vals += (df["off1"] - df["off1"].mean()) * scale_sign
    exp_vals -= df["off2"] * scale_sign
    exp_vals += df["off3"].mean() * scale_sign
    np.testing.assert_allclose(res["energy"].values, exp_vals.values)
    exp_meta: dict[str, Any] = {}
    exp_meta["applied"] = True
    exp_meta["offsets"] = ec.offsets
    assert meta == exp_meta
    # test with explicit params
    ec = EnergyCalibrator(
        config=config,
        loader=get_loader("flash", config=config),
    )
    t_df = dask.dataframe.from_pandas(df.copy(), npartitions=2)
    res, meta = ec.add_offsets(t_df, **params)  # type: ignore
    np.testing.assert_allclose(res["energy"].values, exp_vals.values)
    exp_meta = {}
    exp_meta["applied"] = True
    exp_meta["offsets"] = ec.offsets
    assert meta == exp_meta
    # test with minimal parameters
    ec = EnergyCalibrator(
        config=config,
        loader=get_loader("flash", config=config),
    )
    t_df = dask.dataframe.from_pandas(df.copy(), npartitions=2)
    res, meta = ec.add_offsets(t_df, weights=-1, columns="off1")
    res, meta = ec.add_offsets(res, columns="off1")
    exp_vals = df["energy"].copy()
    np.testing.assert_allclose(res["energy"].values, exp_vals.values)
    exp_meta = {}
    exp_meta["applied"] = True
    exp_meta["offsets"] = ec.offsets
    assert meta == exp_meta


def test_add_offset_raises() -> None:
    """test if add_offset raises the correct errors"""
    cfg_dict: dict[str, Any] = {
        "energy": {
            "calibration": {
                "energy_scale": "kinetic",
            },
            "offsets": {
                "constant": 1,
                "columns": {
                    "off1": {"weight": -1, "preserve_mean": True},
                    "off2": {"weight": -1, "preserve_mean": False},
                    "off3": {"weight": 1, "preserve_mean": False, "reduction": "mean"},
                },
            },
        },
    }

    df = pd.DataFrame(
        {
            "energy": [10, 20, 30, 40, 50, 60],
            "off1": [1, 2, 3, 4, 5, 6],
            "off2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "off3": [10.1, 10.2, 10.3, 10.4, 10.5, 10.6],
        },
    )
    t_df = dask.dataframe.from_pandas(df.copy(), npartitions=2)

    # no energy scale
    with pytest.raises(ValueError):
        cfg = deepcopy(cfg_dict)
        cfg["energy"]["calibration"].pop("energy_scale")
        config = parse_config(config=cfg, folder_config={}, user_config={}, system_config={})
        ec = EnergyCalibrator(config=config, loader=get_loader("flash", config=config))
        _ = ec.add_offsets(t_df)

    # invalid energy scale
    with pytest.raises(ValueError):
        cfg = deepcopy(cfg_dict)
        cfg["energy"]["calibration"]["energy_scale"] = "wrong_value"
        config = parse_config(config=cfg, folder_config={}, user_config={}, system_config={})
        ec = EnergyCalibrator(config=config, loader=get_loader("flash", config=config))
        _ = ec.add_offsets(t_df)

    # invalid sign
    with pytest.raises(TypeError):
        config = parse_config(config=cfg_dict, folder_config={}, user_config={}, system_config={})
        config["energy"]["offsets"]["columns"]["off1"]["weight"] = "wrong_type"
        ec = EnergyCalibrator(config=config, loader=get_loader("flash", config=config))
        _ = ec.add_offsets(t_df)

        # invalid constant
    with pytest.raises(TypeError):
        config = parse_config(config=cfg_dict, folder_config={}, user_config={}, system_config={})
        config["energy"]["offsets"]["constant"] = "wrong_type"
        ec = EnergyCalibrator(config=config, loader=get_loader("flash", config=config))
        _ = ec.add_offsets(t_df)


def test_align_dld_sectors() -> None:
    """test functionality and error handling of align_dld_sectors"""
    cfg_dict: dict[str, Any] = {
        "dataframe": {
            "columns": {
                "tof": "dldTimeSteps",
                "sector_id": "dldSectorId",
            },
            "sector_delays": [-0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35],
        },
    }
    df = pd.DataFrame(
        {
            "dldTimeSteps": [1, 2, 3, 4, 5, 6, 7, 8],
            "dldSectorId": [0, 1, 2, 3, 4, 5, 6, 7],
        },
    )
    # from config
    config = parse_config(config=cfg_dict, folder_config={}, user_config={}, system_config={})
    ec = EnergyCalibrator(config=config, loader=get_loader("flash", config=config))
    t_df = dask.dataframe.from_pandas(df.copy(), npartitions=2)
    res, meta = ec.align_dld_sectors(t_df)
    assert meta["applied"] is True
    assert meta["sector_delays"] == cfg_dict["dataframe"]["sector_delays"]
    np.testing.assert_allclose(
        res["dldTimeSteps"].values,
        np.array([1, 2, 3, 4, 5, 6, 7, 8]) - np.array(cfg_dict["dataframe"]["sector_delays"]),
    )

    # from kwds
    config = parse_config(config={}, folder_config={}, user_config={}, system_config={})
    ec = EnergyCalibrator(config=config, loader=get_loader("flash", config=config))
    t_df = dask.dataframe.from_pandas(df.copy(), npartitions=2)
    res, meta = ec.align_dld_sectors(
        t_df,
        tof_column=cfg_dict["dataframe"]["columns"]["tof"],
        sector_delays=cfg_dict["dataframe"]["sector_delays"],
        sector_id_column="dldSectorId",
    )
    assert meta["applied"] is True
    assert meta["sector_delays"] == cfg_dict["dataframe"]["sector_delays"]
    np.testing.assert_allclose(
        res["dldTimeSteps"].values,
        np.array([1, 2, 3, 4, 5, 6, 7, 8]) - np.array(cfg_dict["dataframe"]["sector_delays"]),
    )
    with pytest.raises(ValueError):
        cfg = deepcopy(cfg_dict)
        cfg["dataframe"].pop("sector_delays")
        config = parse_config(config=cfg, folder_config={}, user_config={}, system_config={})
        ec = EnergyCalibrator(config=config, loader=get_loader("flash", config=config))
        t_df = dask.dataframe.from_pandas(df.copy(), npartitions=2)
        res, meta = ec.align_dld_sectors(t_df)

"""Module tests.calibrator.energy, tests for the sed.calibrator.energy module
"""
import csv
import glob
import itertools
import os
from importlib.util import find_spec
from typing import Any
from typing import Dict

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sed.calibrator.energy import EnergyCalibrator
from sed.core.config import parse_config
from sed.loader.loader_interface import get_loader

package_dir = os.path.dirname(find_spec("sed").origin)
df_folder = package_dir + "/../tests/data/loader/mpes/"
folder = package_dir + "/../tests/data/calibrator/"
files = glob.glob(df_folder + "*.h5")
config = parse_config(
    package_dir + "/config/mpes_example_config.yaml",
    user_config={},
    system_config={},
)

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


def test_bin_data_and_read_biases_from_files():
    """Test binning the data and extracting the bias values from the files"""
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
    default_config = parse_config(config={}, user_config={}, system_config={})
    ec = EnergyCalibrator(
        config=default_config,
        loader=get_loader("mpes", config=default_config),
    )
    with pytest.raises(ValueError):
        ec.bin_data(data_files=files)
    faulty_config = parse_config(
        config={"energy": {"bias_key": "@KTOF:Lens:Sample"}},
        user_config={},
        system_config={},
    )
    ec = EnergyCalibrator(
        config=faulty_config,
        loader=get_loader("mpes", config=faulty_config),
    )
    with pytest.raises(ValueError):
        ec.bin_data(data_files=files)


def test_energy_calibrator_from_arrays_norm():
    """Test loading the energy, bias and tof traces into the class"""
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


def test_feature_extract():
    """Test generating the ranges, and extracting the features"""
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


def test_adjust_ranges():
    """Test the interactive function for adjusting the feature ranges"""
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


energy_scales = ["kinetic", "binding"]
calibration_methods = ["lmfit", "lstsq", "lsqr"]


@pytest.mark.parametrize(
    "energy_scale, calibration_method",
    itertools.product(energy_scales, calibration_methods),
)
def test_calibrate_append(energy_scale: str, calibration_method: str):
    """Test if the calibration routines generate the correct slope of energy vs. tof,
    and the application to the data frame.

    Args:
        energy_scale (str): tpye of energy scaling
        calibration_method (str): method used for ralibration
    """
    loader = get_loader(loader_name="mpes", config=config)
    df, _ = loader.read_dataframe(folders=df_folder, collect_metadata=False)
    ec = EnergyCalibrator(config=config, loader=loader)
    ec.load_data(biases=biases, traces=traces, tof=tof)
    ec.normalize()
    rng = (66100, 67000)
    ref_id = 5
    ec.add_ranges(ranges=rng, ref_id=ref_id)
    ec.feature_extract()
    refid = 4
    e_ref = -0.5
    calibdict = ec.calibrate(
        ref_energy=e_ref,
        ref_id=refid,
        energy_scale=energy_scale,
        method=calibration_method,
    )
    df, metadata = ec.append_energy_axis(df)
    assert config["dataframe"]["energy_column"] in df.columns
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


calib_types = ["fit", "poly"]
calib_dicts = [{"d": 1, "t0": 0, "E0": 0}, {"coeffs": [1, 2, 3], "E0": 0}]


@pytest.mark.parametrize(
    "calib_type, calib_dict",
    zip(calib_types, calib_dicts),
)
def test_apply_correction_from_dict_kwds(calib_type: str, calib_dict: dict):
    """Function to test if the energy calibration is correctly applied using a dict or
    kwd parameters.

    Args:
        calib_type (str): type of calibration.
        calib_dict (dict): Dictionary with calibration parameters.
    """
    loader = get_loader(loader_name="mpes", config=config)
    # from dict
    df, _ = loader.read_dataframe(folders=df_folder, collect_metadata=False)
    ec = EnergyCalibrator(config=config, loader=loader)
    df, metadata = ec.append_energy_axis(df, calibration=calib_dict)
    assert config["dataframe"]["energy_column"] in df.columns

    for key in calib_dict:
        np.testing.assert_equal(metadata["calibration"][key], calib_dict[key])
    assert metadata["calibration"]["calib_type"] == calib_type

    # from kwds
    df, _ = loader.read_dataframe(folders=df_folder, collect_metadata=False)
    ec = EnergyCalibrator(config=config, loader=loader)
    df, metadata = ec.append_energy_axis(df, **calib_dict)
    assert config["dataframe"]["energy_column"] in df.columns

    for key in calib_dict:
        np.testing.assert_equal(metadata["calibration"][key], calib_dict[key])
    assert metadata["calibration"]["calib_type"] == calib_type


amplitude = 2.5  # pylint: disable=invalid-name
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
def test_energy_correction(correction_type: str, correction_kwd: dict):
    """Function to test if all energy correction functions generate symmetric curves
    with the maximum at the cetner x/y.

    Args:
        correction_type (str): type of correction to test
        correction_kwd (dict): parameters to pass to the function
    """
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
    t = df[config["dataframe"]["corrected_tof_column"]]
    assert t[0] == t[2]
    assert t[0] < t[1]
    assert t[3] == t[5]
    assert t[3] < t[4]
    assert t[1] == t[4]

    for key, value in ec.correction.items():
        np.testing.assert_equal(
            metadata["correction"][key],
            value,
        )


@pytest.mark.parametrize(
    "correction_type, correction_kwd",
    zip(correction_types, correction_kwds),
)
def test_energy_correction_from_dict_kwds(
    correction_type: str,
    correction_kwd: dict,
):
    """Function to test if the energy correction is correctly applied using a dict or
    kwd parameters.

    Args:
        correction_type (str): type of correction to test
        correction_kwd (dict): parameters to pass to the function
    """
    # from dict
    sample_df = pd.DataFrame(sample, columns=columns)
    ec = EnergyCalibrator(
        config=config,
        loader=get_loader("mpes", config=config),
    )
    correction_dict: Dict[str, Any] = {
        "correction_type": correction_type,
        "amplitude": amplitude,
        "center": center,
        **correction_kwd,
    }
    df, metadata = ec.apply_energy_correction(
        sample_df,
        correction=correction_dict,
    )
    t = df[config["dataframe"]["corrected_tof_column"]]
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
    t = df[config["dataframe"]["corrected_tof_column"]]
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

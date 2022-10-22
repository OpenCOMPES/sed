"""Module tests.calibrator.energy, tests for the sed.calibrator.energy module
"""
import csv
import glob
import itertools
import os
from importlib.util import find_spec

import numpy as np
import pytest

from sed.calibrator.energy import EnergyCalibrator
from sed.config.settings import parse_config
from sed.loader.mpes import MpesLoader

package_dir = os.path.dirname(find_spec("sed").origin)
df_folder = package_dir + "/../tests/data/loader/"
folder = package_dir + "/../tests/data/calibrator/"
files = glob.glob(df_folder + "*.h5")
config = parse_config(package_dir + "/../tests/data/config/config.yaml")

traces = []
with open(folder + "traces.csv", newline="") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        traces.append(row)
with open(folder + "tof.csv", newline="") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    tof = next(reader)
with open(folder + "biases.csv", newline="") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    biases = next(reader)

traces = np.asarray(traces).T
tof = np.asarray(tof)
biases = np.asarray(biases)


def test_bin_data_and_read_biases_from_files():
    """Test binning the data and extracting the bias values from the files"""
    ec = EnergyCalibrator(config=config)  # pylint: disable=invalid-name
    ec.bin_data(data_files=files)
    assert ec.traces.ndim == 2
    assert ec.traces.shape == (2, 1000)
    assert ec.tof.ndim == 1
    assert ec.biases.ndim == 1
    assert len(ec.biases) == 2


def test_energy_calibrator_from_arrays_norm():
    """Test loading the energy, bias and tof traces into the class"""
    # df = MpesLoader(config=config).read_dataframe(folder=df_folder)
    ec = EnergyCalibrator(config=config)  # pylint: disable=invalid-name
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
    ec = EnergyCalibrator(config=config)  # pylint: disable=invalid-name
    ec.load_data(biases=biases, traces=traces_rand, tof=tof)
    ec.add_features(ranges=rng, ref_id=ref_id)
    for pos, feat_rng in zip(rand, ec.featranges):
        assert feat_rng[0] < (tof[1] - tof[0]) * pos + 65000 < feat_rng[1]
    ec.feature_extract()
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
    df = MpesLoader(config=config).read_dataframe(folder=df_folder)
    ec = EnergyCalibrator(config=config)  # pylint: disable=invalid-name
    ec.load_data(biases=biases, traces=traces, tof=tof)
    ec.normalize()
    rng = (66100, 67000)
    ref_id = 5
    ec.add_features(ranges=rng, ref_id=ref_id)
    ec.feature_extract()
    refid = 4
    e_ref = -0.5
    calibdict = ec.calibrate(
        ref_energy=e_ref,
        ref_id=refid,
        energy_scale=energy_scale,
        method=calibration_method,
    )
    df = ec.append_energy_axis(df)
    assert "E" in df.columns
    axis = calibdict["axis"]
    diff = np.diff(axis)
    if energy_scale == "kinetic":
        assert np.all(diff < 0)
    else:
        assert np.all(diff > 0)

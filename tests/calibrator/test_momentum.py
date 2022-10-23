"""Module tests.calibrator.momentum, tests for the sed.calibrator.momentum module
"""
import csv
import glob
import os
from importlib.util import find_spec

import numpy as np

from sed.calibrator.momentum import MomentumCorrector
from sed.config.settings import parse_config
from sed.core import SedProcessor
from sed.loader.mpes import MpesLoader

package_dir = os.path.dirname(find_spec("sed").origin)
df_folder = package_dir + "/../tests/data/loader/"
folder = package_dir + "/../tests/data/calibrator/"
files = glob.glob(df_folder + "*.h5")
config = parse_config(package_dir + "/../tests/data/config/config.yaml")

momentum_map_list = []
with open(
    folder + "momentum_map.csv",
    newline="",
    encoding="utf-8",
) as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        momentum_map_list.append(row)
momentum_map = np.asarray(momentum_map_list).T


def test_bin_data_and_slice_image():
    """Test binning the data and slicing of the image"""
    sp = SedProcessor(config=config)  # pylint: disable=invalid-name
    sp.load(files=files)
    sp.bin_and_load_momentum_calibration(plane=33, width=10, apply=True)
    assert sp.mc.slice.shape == (512, 512)


def test_feature_extract():
    """Testextracting the feature from a 2D slice"""
    mc = MomentumCorrector(config=config)  # pylint: disable=invalid-name
    mc.load_data(
        data=momentum_map,
        bin_ranges=[(-256, 1792), (-256, 1792)],
        rotsym=6,
    )
    mc.feature_extract(fwhm=10, sigma=12, sigma_radius=4)
    assert len(mc.pcent) == 2
    assert len(mc.pouter_ord) == 6


def test_splinewarp():
    """Test the generation of the splinewarp etimate."""
    mc = MomentumCorrector(config=config)  # pylint: disable=invalid-name
    mc.load_data(
        data=momentum_map,
        bin_ranges=[(-256, 1792), (-256, 1792)],
        rotsym=6,
    )
    features = np.array(
        [
            [203.2, 341.96],
            [299.16, 345.32],
            [350.25, 243.70],
            [304.38, 149.88],
            [199.52, 152.48],
            [154.28, 242.27],
            [248.29, 248.62],
        ],
    )
    mc.add_features(peaks=features)
    mc.spline_warp_estimate()
    assert mc.cdeform_field.shape == mc.rdeform_field.shape == mc.image.shape
    assert len(mc.ptargs) == len(mc.prefs)


def test_pose_correction():
    """Test the adjustment of the pose correction."""
    mc = MomentumCorrector(config=config)  # pylint: disable=invalid-name
    mc.load_data(
        data=momentum_map,
        bin_ranges=[(-256, 1792), (-256, 1792)],
        rotsym=6,
    )
    features = np.array(
        [
            [203.2, 341.96],
            [299.16, 345.32],
            [350.25, 243.70],
            [304.38, 149.88],
            [199.52, 152.48],
            [154.28, 242.27],
            [248.29, 248.62],
        ],
    )
    mc.add_features(peaks=features)
    mc.spline_warp_estimate()
    dfield = np.array([mc.cdeform_field, mc.rdeform_field])
    mc.pose_adjustment(scale=1.2, xtrans=8, ytrans=7, angle=-4, apply=True)
    assert np.all(np.array([mc.cdeform_field, mc.rdeform_field]) != dfield)


def test_apply_correction():
    """Test the application of the distortion correction to the dataframe."""
    df = MpesLoader(config=config).read_dataframe(folder=df_folder)
    mc = MomentumCorrector(config=config)  # pylint: disable=invalid-name
    mc.load_data(
        data=momentum_map,
        bin_ranges=[(-256, 1792), (-256, 1792)],
        rotsym=6,
    )
    features = np.array(
        [
            [203.2, 341.96],
            [299.16, 345.32],
            [350.25, 243.70],
            [304.38, 149.88],
            [199.52, 152.48],
            [154.28, 242.27],
            [248.29, 248.62],
        ],
    )
    mc.add_features(peaks=features)
    mc.spline_warp_estimate()
    df = mc.apply_distortion_correction(df=df)
    assert "Xm" in df.columns
    assert "Ym" in df.columns


def test_momentum_calibration_equiscale():
    """Test the calibration using one point and the k-distance,
    and application to the dataframe.
    """
    df = MpesLoader(config=config).read_dataframe(folder=df_folder)
    mc = MomentumCorrector(config=config)  # pylint: disable=invalid-name
    mc.load_data(data=momentum_map, bin_ranges=[(-256, 1792), (-256, 1792)])
    point_a = [308, 345]
    point_b = [256, 256]
    k_distance = 4 / 3 * np.pi / 3.28
    mc.calibrate(
        image=momentum_map,
        point_a=point_a,
        point_b=point_b,
        k_distance=k_distance,
        equiscale=True,
    )
    mc.append_k_axis(df, x_column="X", y_column="Y")
    assert "kx" in df.columns
    assert "ky" in df.columns


def test_momentum_calibration_two_points():
    """Test the calibration using two k-points, and application to the dataframe."""
    df = MpesLoader(config=config).read_dataframe(folder=df_folder)
    mc = MomentumCorrector(config=config)  # pylint: disable=invalid-name
    mc.load_data(data=momentum_map, bin_ranges=[(-256, 1792), (-256, 1792)])
    point_a = [360, 256]
    point_b = [256, 360]
    k_coord_a = 4 / 3 * np.pi / 3.28 * np.array([1, 0])
    k_coord_b = 4 / 3 * np.pi / 3.28 * np.array([0, 1])
    mc.calibrate(
        image=momentum_map,
        point_a=point_a,
        point_b=point_b,
        k_coord_a=k_coord_a,
        k_coord_b=k_coord_b,
        equiscale=False,
    )
    mc.append_k_axis(df, x_column="X", y_column="Y")
    assert "kx" in df.columns
    assert "ky" in df.columns

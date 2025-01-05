"""Module tests.calibrator.momentum, tests for the sed.calibrator.momentum module
"""
from __future__ import annotations

import csv
import glob
import os
from typing import Any

import numpy as np
import pytest

from sed.calibrator.momentum import MomentumCorrector
from sed.core import SedProcessor
from sed.core.config import parse_config
from sed.loader.loader_interface import get_loader

test_dir = os.path.join(os.path.dirname(__file__), "..")
df_folder = test_dir + "/data/loader/mpes/"
folder = test_dir + "/data/calibrator/"
files = glob.glob(df_folder + "*.h5")

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


def test_bin_data_and_slice_image() -> None:
    """Test binning the data and slicing of the image"""
    config = parse_config(
        config={"core": {"loader": "mpes"}},
        folder_config={},
        user_config={},
        system_config={},
    )
    sed_processor = SedProcessor(config=config, system_config={})
    sed_processor.load(files=files)
    sed_processor.bin_and_load_momentum_calibration(
        plane=33,
        width=10,
        apply=True,
    )
    assert sed_processor.mc.slice.shape == (512, 512)


def test_feature_extract() -> None:
    """Test extracting the feature from a 2D slice"""
    config = parse_config(
        config={"core": {"loader": "mpes"}},
        folder_config={},
        user_config={},
        system_config={},
    )
    mc = MomentumCorrector(config=config)
    mc.load_data(
        data=momentum_map,
        bin_ranges=[(-256, 1792), (-256, 1792)],
    )
    mc.feature_extract(fwhm=10, sigma=12, sigma_radius=4, rotsym=6)
    assert len(mc.pcent) == 2
    assert len(mc.pouter_ord) == 6

    # illegal keywords
    with pytest.raises(TypeError):
        mc.feature_extract(features=np.ndarray([1, 2]), illegal_kwd=True)


@pytest.mark.parametrize(
    "include_center",
    [True, False],
)
def test_splinewarp(include_center: bool) -> None:
    """Test the generation of the splinewarp estimate.

    Args:
        include_center (bool): Option to include the center point.
    """
    config = parse_config(
        config={"core": {"loader": "mpes"}},
        folder_config={},
        user_config={},
        system_config={},
    )
    mc = MomentumCorrector(config=config)
    mc.load_data(
        data=momentum_map,
        bin_ranges=[(-256, 1792), (-256, 1792)],
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
    if not include_center:
        features = features[0:-1]
    mc.add_features(features=features, rotsym=6)
    mc.spline_warp_estimate(use_center=include_center)
    assert mc.cdeform_field.shape == mc.rdeform_field.shape == mc.image.shape
    assert len(mc.ptargs) == len(mc.prefs)


def test_ascale() -> None:
    """Test the generation of the splinewarp estimate with ascale parameter."""
    config = parse_config(
        config={"core": {"loader": "mpes"}},
        folder_config={},
        user_config={},
        system_config={},
    )
    mc = MomentumCorrector(config=config)
    mc.load_data(
        data=momentum_map,
        bin_ranges=[(-256, 1792), (-256, 1792)],
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
    mc.add_features(features=features, rotsym=6)
    with pytest.raises(ValueError):
        mc.spline_warp_estimate(ascale=1.3)
    with pytest.raises(ValueError):
        mc.spline_warp_estimate(ascale=[1.3, 1, 1.3, 1])
    with pytest.raises(TypeError):
        mc.spline_warp_estimate(ascale="invalid type")  # type:ignore
    mc.spline_warp_estimate(ascale=[1.3, 1, 1.3, 1, 1.3, 1])
    assert mc.cdeform_field.shape == mc.rdeform_field.shape == mc.image.shape
    assert len(mc.ptargs) == len(mc.prefs)
    # test single value case
    with pytest.raises(ValueError):
        mc.add_features(features=features, rotsym=4)
    mc.add_features(features=features[:5, :], rotsym=4)
    mc.spline_warp_estimate(ascale=1.3)


def test_pose_correction() -> None:
    """Test the adjustment of the pose correction."""
    config = parse_config(
        config={"core": {"loader": "mpes"}},
        folder_config={},
        user_config={},
        system_config={},
    )
    mc = MomentumCorrector(config=config)
    mc.load_data(
        data=momentum_map,
        bin_ranges=[(-256, 1792), (-256, 1792)],
    )
    mc.reset_deformation()
    dfield = np.array([mc.cdeform_field, mc.rdeform_field])
    mc.pose_adjustment(scale=1.2, xtrans=8, ytrans=7, angle=-4, apply=True)
    assert np.all(np.array([mc.cdeform_field, mc.rdeform_field]) != dfield)

    # Illegal keywords:
    with pytest.raises(TypeError):
        mc.reset_deformation(illegal_kwd=True)


def test_apply_correction() -> None:
    """Test the application of the distortion correction to the dataframe."""
    config = parse_config(
        config={"core": {"loader": "mpes"}},
        folder_config={},
        user_config={},
        system_config={},
    )
    df, _, _ = get_loader(loader_name="mpes", config=config).read_dataframe(
        folders=df_folder,
        collect_metadata=False,
    )
    mc = MomentumCorrector(config=config)
    mc.load_data(
        data=momentum_map,
        bin_ranges=[(-256, 1792), (-256, 1792)],
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
    mc.add_features(features=features, rotsym=6)
    mc.spline_warp_estimate()
    df, metadata = mc.apply_corrections(df=df)
    assert "Xm" in df.columns
    assert "Ym" in df.columns
    assert metadata["correction"]["applied"] is True
    np.testing.assert_equal(metadata["correction"]["reference_points"], features)
    assert metadata["correction"]["cdeform_field"].shape == momentum_map.shape
    assert metadata["correction"]["rdeform_field"].shape == momentum_map.shape


transformations_list = [
    {"xtrans": np.random.randint(1, 50)},
    {"ytrans": np.random.randint(1, 50)},
    {"angle": np.random.randint(1, 50)},
    {
        "xtrans": np.random.randint(1, 50),
        "ytrans": np.random.randint(1, 50),
    },
    {
        "xtrans": np.random.randint(1, 50),
        "angle": np.random.randint(1, 50),
    },
    {
        "ytrans": np.random.randint(1, 50),
        "angle": np.random.randint(1, 50),
    },
    {
        "xtrans": np.random.randint(1, 50),
        "ytrans": np.random.randint(1, 50),
        "angle": np.random.randint(1, 50),
    },
]
depends_on_list = [
    {
        "root": "/entry/process/registration/transformations/trans_x",
        "axes": {"trans_x": "."},
    },
    {
        "root": "/entry/process/registration/transformations/trans_y",
        "axes": {"trans_y": "."},
    },
    {
        "root": "/entry/process/registration/transformations/rot_z",
        "axes": {"rot_z": "."},
    },
    {
        "root": "/entry/process/registration/transformations/trans_y",
        "axes": {
            "trans_x": ".",
            "trans_y": "/entry/process/registration/transformations/trans_x",
        },
    },
    {
        "root": "/entry/process/registration/transformations/rot_z",
        "axes": {
            "trans_x": ".",
            "rot_z": "/entry/process/registration/transformations/trans_x",
        },
    },
    {
        "root": "/entry/process/registration/transformations/rot_z",
        "axes": {
            "trans_y": ".",
            "rot_z": "/entry/process/registration/transformations/trans_y",
        },
    },
    {
        "root": "/entry/process/registration/transformations/rot_z",
        "axes": {
            "trans_x": ".",
            "trans_y": "/entry/process/registration/transformations/trans_x",
            "rot_z": "/entry/process/registration/transformations/trans_y",
        },
    },
]


@pytest.mark.parametrize(
    "transformations, depends_on",
    zip(transformations_list, depends_on_list),
)
def test_apply_registration(
    transformations: dict[Any, Any],
    depends_on: dict[Any, Any],
) -> None:
    """Test the application of the distortion correction to the dataframe."""
    config = parse_config(
        config={"core": {"loader": "mpes"}},
        folder_config={},
        user_config={},
        system_config={},
    )
    df, _, _ = get_loader(loader_name="mpes", config=config).read_dataframe(
        folders=df_folder,
        collect_metadata=False,
    )
    mc = MomentumCorrector(config=config)
    mc.load_data(
        data=momentum_map,
        bin_ranges=[(-256, 1792), (-256, 1792)],
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
    # store dummy deformation
    mc.reset_deformation()
    dummy_inv_dfield = np.asarray(
        [
            np.asarray([np.arange(0, 2048) for _ in range(2048)]),
            np.asarray([np.arange(0, 2048) for _ in range(2048)]).T,
        ],
    )
    mc.inverse_dfield = dummy_inv_dfield
    mc.add_features(features=features, rotsym=6)
    mc.spline_warp_estimate()
    mc.pose_adjustment(**transformations, apply=True)
    # disable re-calculation of inverse dfield to save time, as we are just testing meta data here
    mc.dfield_updated = False
    df, metadata = mc.apply_corrections(df=df)
    assert "Xm" in df.columns
    assert "Ym" in df.columns
    assert metadata["registration"]["applied"] is True
    assert metadata["registration"]["depends_on"] == depends_on["root"]
    for key, value in transformations.items():
        if key == "xtrans":
            assert metadata["registration"]["trans_x"]["value"] == value
            assert (
                metadata["registration"]["trans_x"]["depends_on"] == depends_on["axes"]["trans_x"]
            )
            assert metadata["registration"]["trans_x"]["type"] == "translation"
        if key == "ytrans":
            assert metadata["registration"]["trans_y"]["value"] == value
            assert (
                metadata["registration"]["trans_y"]["depends_on"] == depends_on["axes"]["trans_y"]
            )
            assert metadata["registration"]["trans_y"]["type"] == "translation"
        if key == "angle":
            assert metadata["registration"]["rot_z"]["value"] == value
            assert metadata["registration"]["rot_z"]["depends_on"] == depends_on["axes"]["rot_z"]
            assert metadata["registration"]["rot_z"]["type"] == "rotation"
            np.testing.assert_equal(
                metadata["registration"]["rot_z"]["offset"][0:2],
                metadata["registration"]["center"],
            )

    # illegal keywords:
    with pytest.raises(TypeError):
        mc.pose_adjustment(illegal_kwd=True)


def test_momentum_calibration_equiscale() -> None:
    """Test the calibration using one point and the k-distance,
    and application to the dataframe.
    """
    config = parse_config(
        config={"core": {"loader": "mpes"}},
        folder_config={},
        user_config={},
        system_config={},
    )
    df, _, _ = get_loader(loader_name="mpes", config=config).read_dataframe(
        folders=df_folder,
        collect_metadata=False,
    )
    mc = MomentumCorrector(config=config)
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
    df, metadata = mc.append_k_axis(df, x_column="X", y_column="Y")
    assert "kx" in df.columns
    assert "ky" in df.columns
    for key, value in mc.calibration.items():
        np.testing.assert_equal(metadata["calibration"][key], value)

    # illegal keywords:
    with pytest.raises(TypeError):
        mc.append_k_axis(df, illegal_kwd=True)


def test_momentum_calibration_two_points() -> None:
    """Test the calibration using two k-points, and application to the dataframe."""
    config = parse_config(
        config={"core": {"loader": "mpes"}},
        folder_config={},
        user_config={},
        system_config={},
    )
    df, _, _ = get_loader(loader_name="mpes", config=config).read_dataframe(
        folders=df_folder,
        collect_metadata=False,
    )
    mc = MomentumCorrector(config=config)
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
    df, metadata = mc.append_k_axis(df, x_column="X", y_column="Y")
    assert "kx" in df.columns
    assert "ky" in df.columns
    for key, value in mc.calibration.items():
        np.testing.assert_equal(
            metadata["calibration"][key],
            value,
        )
    # Test with passing calibration parameters
    calibration = mc.calibration.copy()
    calibration.pop("creation_date")
    calibration.pop("grid")
    calibration.pop("extent")
    calibration.pop("kx_axis")
    calibration.pop("ky_axis")
    df, _, _ = get_loader(loader_name="mpes", config=config).read_dataframe(
        folders=df_folder,
        collect_metadata=False,
    )
    mc = MomentumCorrector(config=config)
    df, metadata = mc.append_k_axis(df, **calibration)
    assert "kx" in df.columns
    assert "ky" in df.columns
    for key, value in mc.calibration.items():
        np.testing.assert_equal(
            metadata["calibration"][key],
            value,
        )

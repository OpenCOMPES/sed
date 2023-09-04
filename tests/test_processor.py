"""Module tests.processor, tests for the sed.core.processor module
"""
import glob
import os
from importlib.util import find_spec

import numpy as np
import pytest

from sed import SedProcessor
from sed.core.config import parse_config
from sed.loader.loader_interface import get_loader

#  pylint: disable=duplicate-code
package_dir = os.path.dirname(find_spec("sed").origin)
df_folder = package_dir + "/../tests/data/loader/generic/"
folder = package_dir + "/../tests/data/calibrator/"
files = glob.glob(df_folder + "*.parquet")
loader = get_loader(loader_name="generic")


def test_processor_from_dataframe():
    """Test generation of the processor from a dataframe object"""
    config = parse_config(
        config={"core": {"loader": "generic"}},
        folder_config={},
        user_config={},
        system_config={},
    )
    dataframe, _ = loader.read_dataframe(files=files)
    processor = SedProcessor(
        dataframe=dataframe,
        config=config,
        folder_config={},
        user_config={},
        system_config={},
    )
    for column in dataframe.columns:
        assert (dataframe[column].compute() == processor.dataframe[column].compute()).all()


def test_processor_from_files():
    """Test generation of the processor from a list of files"""
    config = parse_config(
        config={"core": {"loader": "generic"}},
        folder_config={},
        user_config={},
        system_config={},
    )
    dataframe, _ = loader.read_dataframe(files=files)
    processor = SedProcessor(
        files=files,
        config=config,
        folder_config={},
        user_config={},
        system_config={},
    )
    for column in dataframe.columns:
        assert (dataframe[column].compute() == processor.dataframe[column].compute()).all()


def test_processor_from_folders():
    """Test generation of the processor from a folder"""
    config = parse_config(
        config={"core": {"loader": "generic"}},
        folder_config={},
        user_config={},
        system_config={},
    )
    dataframe, _ = loader.read_dataframe(files=files)
    processor = SedProcessor(
        folder=df_folder,
        config=config,
        folder_config={},
        user_config={},
        system_config={},
    )
    for column in dataframe.columns:
        assert (dataframe[column].compute() == processor.dataframe[column].compute()).all()


feature4 = np.array([[203.2, 341.96], [299.16, 345.32], [304.38, 149.88], [199.52, 152.48]])
feature5 = np.array(
    [[203.2, 341.96], [299.16, 345.32], [304.38, 149.88], [199.52, 152.48], [248.29, 248.62]],
)
feature6 = np.array(
    [
        [203.2, 341.96],
        [299.16, 345.32],
        [350.25, 243.70],
        [304.38, 149.88],
        [199.52, 152.48],
        [154.28, 242.27],
    ],
)
feature7 = np.array(
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
feature_list = [feature4, feature5, feature6, feature7]

adjust_params = {
    "scale": np.random.randint(1, 10) / 10 + 0.5,
    "xtrans": np.random.randint(1, 50),
    "ytrans": np.random.randint(1, 50),
    "angle": np.random.randint(1, 50),
}


@pytest.mark.parametrize(
    "features",
    feature_list,
)
def test_momentum_correction_workflow(features):
    """Test for the momentum correction workflow"""
    config = parse_config(
        config={"core": {"loader": "generic"}},
        folder_config={},
        user_config={},
        system_config={},
    )
    processor = SedProcessor(
        folder=df_folder,
        config=config,
        folder_config={},
        user_config={},
        system_config={},
    )
    processor.bin_and_load_momentum_calibration(apply=True)
    assert processor.mc.slice is not None
    assert processor.mc.pouter is None
    if len(features) == 5 or len(features) == 7:
        rotsym = len(features) - 1
        include_center = True
    else:
        rotsym = len(features)
        include_center = False
    processor.define_features(
        features=features,
        rotation_symmetry=rotsym,
        include_center=include_center,
        apply=True,
    )
    assert len(processor.mc.pouter_ord) == rotsym
    processor.generate_splinewarp(use_center=include_center)
    processor.save_splinewarp(filename=f"sed_config{len(features)}.yaml", overwrite=True)
    pouter_ord = processor.mc.pouter_ord
    cdeform_field = processor.mc.cdeform_field
    rdeform_field = processor.mc.rdeform_field
    # load features
    processor = SedProcessor(
        folder=df_folder,
        config=config,
        folder_config=f"sed_config{len(features)}.yaml",
        user_config={},
        system_config={},
    )
    processor.generate_splinewarp()
    assert len(processor.mc.pouter_ord) == rotsym
    np.testing.assert_allclose(processor.mc.pouter_ord, pouter_ord)
    np.testing.assert_allclose(processor.mc.cdeform_field, cdeform_field)
    np.testing.assert_allclose(processor.mc.rdeform_field, rdeform_field)


def test_pose_adjustment():
    """Test for the pose correction and application of momentum correction workflow"""
    config = parse_config(
        config={"core": {"loader": "generic"}},
        folder_config={},
        user_config={},
        system_config={},
    )
    processor = SedProcessor(
        folder=df_folder,
        config=config,
        folder_config={},
        user_config={},
        system_config={},
    )
    with pytest.raises(ValueError):
        processor.pose_adjustment(**adjust_params, use_correction=False, apply=True)

    processor.bin_and_load_momentum_calibration(apply=True)
    # test pose adjustment
    processor.pose_adjustment(**adjust_params, use_correction=False, apply=True)
    processor = SedProcessor(
        folder=df_folder,
        config=config,
        folder_config="sed_config7.yaml",
        user_config={},
        system_config={},
    )
    processor.bin_and_load_momentum_calibration(apply=True)
    processor.pose_adjustment(**adjust_params, apply=True)
    processor.apply_momentum_correction()
    assert "Xm" in processor.dataframe.columns
    assert "Ym" in processor.dataframe.columns


def test_compute():
    """Test binning of final result"""
    config = parse_config(
        config={"core": {"loader": "generic"}},
        folder_config={},
        user_config={},
        system_config={},
    )
    processor = SedProcessor(
        folder=df_folder,
        config=config,
        folder_config={},
        user_config={},
        system_config={},
    )
    bins = [10, 10, 10, 10]
    axes = ["X", "Y", "t", "ADC"]
    ranges = [[0, 2048], [0, 2048], [0, 200000], [0, 50000]]
    result = processor.compute(bins=bins, axes=axes, ranges=ranges, df_partitions=5)
    assert result.data.shape == (10, 10, 10, 10)
    assert result.data.sum(axis=(0, 1, 2, 3)) > 0

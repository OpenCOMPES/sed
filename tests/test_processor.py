"""Module tests.processor, tests for the sed.core.processor module
"""
import glob
import os
import tempfile
from importlib.util import find_spec

import dask.dataframe as ddf
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
runs = ["43878", "43878"]
loader = get_loader(loader_name="generic")
source_folder = package_dir + "/../"
dest_folder = tempfile.mkdtemp()
gid = os.getgid()


def test_processor_from_dataframe():
    """Test generation of the processor from a dataframe object"""
    config = {"core": {"loader": "generic"}}
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
    config = {"core": {"loader": "generic"}}
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
    config = {"core": {"loader": "generic"}}
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


def test_processor_from_runs():
    """Test generation of the processor from runs"""
    config = df_folder + "../flash/config.yaml"
    processor = SedProcessor(
        folder=df_folder + "../flash/",
        config=config,
        runs=runs,
        folder_config={},
        user_config={},
        system_config={},
    )
    assert "dldPosX" in processor.dataframe.columns


def test_additional_parameter_to_loader():
    """Test if additinal keyword parameter can be passed to the loader from the
    Processor initialiuzation.
    """
    config = {"core": {"loader": "generic"}}
    processor = SedProcessor(
        folder=df_folder,
        ftype="json",
        config=config,
        folder_config={},
        user_config={},
        system_config={},
    )
    assert processor.files[0].find("json") > -1


def test_repr():
    """test the ___repr___ method"""
    config = {"core": {"loader": "generic"}}
    processor = SedProcessor(
        config=config,
        folder_config={},
        user_config={},
        system_config={},
    )
    processor_str = str(processor)
    assert processor_str.find("No Data loaded") > 0
    with pytest.raises(ValueError):
        processor.load()
    processor.load(files=files, metadata={"test": {"key1": "value1"}})
    processor_str = str(processor)
    assert processor_str.find("ADC") > 0
    assert processor_str.find("key1") > 0


def test_attributes_setters():
    """Test class attributes and setters."""
    config = {"core": {"loader": "generic"}}
    processor = SedProcessor(
        config=config,
        folder_config={},
        user_config={},
        system_config={},
    )
    processor.load(files=files, metadata={"test": {"key1": "value1"}})
    dataframe = processor.dataframe
    assert isinstance(dataframe, ddf.DataFrame)
    dataframe["X"] = dataframe["Y"]
    processor.dataframe = dataframe
    np.testing.assert_allclose(
        processor.dataframe["X"].compute(),
        processor.dataframe["Y"].compute(),
    )
    metadata = processor.attributes
    assert isinstance(metadata, dict)
    assert "test" in metadata.keys()
    processor.add_attribute({"key2": 5}, name="test2")
    assert processor.attributes["test2"]["key2"] == 5
    assert processor.config["core"]["loader"] == "generic"
    assert len(processor.files) == 2


def test_copy_tool():
    """Test the copy tool functionality in the processor"""
    config = {"core": {"loader": "generic", "use_copy_tool": True}}
    processor = SedProcessor(
        config=config,
        folder_config={},
        user_config={},
        system_config={},
    )
    assert processor.use_copy_tool is False
    config = {
        "core": {
            "loader": "generic",
            "use_copy_tool": True,
            "copy_tool_source": source_folder,
            "copy_tool_dest": dest_folder,
            "copy_tool_kwds": {"gid": os.getgid()},
        },
    }
    processor = SedProcessor(
        config=config,
        folder_config={},
        user_config={},
        system_config={},
    )
    assert processor.use_copy_tool is True
    processor.load(files=files)
    assert processor.files[0].find(dest_folder) > -1


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


point_a = [308, 345]
k_distance = 4 / 3 * np.pi / 3.28
k_coord_a = [k_distance * 0.3, k_distance * 0.8]


def test_calibrate_momentum_axes():
    """Test the calibration of the momentum axes"""
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
        processor.calibrate_momentum_axes(apply=True)
    processor.bin_and_load_momentum_calibration(apply=True)
    with pytest.raises(AssertionError):
        processor.calibrate_momentum_axes(point_a=point_a, apply=True)
    processor.calibrate_momentum_axes(point_a=point_a, k_distance=k_distance, apply=True)
    assert processor.mc.calibration["kx_scale"] == processor.mc.calibration["ky_scale"]
    with pytest.raises(AssertionError):
        processor.calibrate_momentum_axes(point_a=point_a, k_coord_a=k_coord_a, apply=True)
    with pytest.raises(AssertionError):
        processor.calibrate_momentum_axes(point_a=point_a, equiscale=False, apply=True)
    processor.calibrate_momentum_axes(
        point_a=point_a,
        k_coord_a=k_coord_a,
        equiscale=False,
        apply=True,
    )
    assert processor.mc.calibration["kx_scale"] != processor.mc.calibration["ky_scale"]
    processor.save_momentum_calibration()
    processor = SedProcessor(
        folder=df_folder,
        config=config,
        user_config={},
        system_config={},
    )
    processor.apply_momentum_calibration()
    assert (
        processor.config["momentum"]["calibration"]["kx_scale"]
        != processor.config["momentum"]["calibration"]["ky_scale"]
    )
    assert "kx" in processor.dataframe.columns
    assert "ky" in processor.dataframe.columns


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

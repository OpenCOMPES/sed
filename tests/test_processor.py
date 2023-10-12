"""Module tests.processor, tests for the sed.core.processor module
"""
import csv
import glob
import itertools
import os
import tempfile
from importlib.util import find_spec
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

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
    # cleanup flash inermediaries
    _, parquet_data_dir = processor.loader.initialize_paths()
    for file in os.listdir(Path(parquet_data_dir, "buffer")):
        os.remove(Path(parquet_data_dir, "buffer", file))


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
    with pytest.raises(ValueError):
        processor.dataframe = dataframe["X"]
    dataframe["X"] = dataframe["Y"]
    processor.dataframe = dataframe
    np.testing.assert_allclose(
        processor.dataframe["X"].compute(),
        processor.dataframe["Y"].compute(),
    )
    processor_metadata = processor.attributes
    assert isinstance(processor_metadata, dict)
    assert "test" in processor_metadata.keys()
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
def test_momentum_correction_workflow(features: np.ndarray):
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
    processor.save_splinewarp(
        filename=f"sed_config_momentum_correction{len(features)}.yaml",
        overwrite=True,
    )
    pouter_ord = processor.mc.pouter_ord
    cdeform_field = processor.mc.cdeform_field
    rdeform_field = processor.mc.rdeform_field
    # load features
    processor = SedProcessor(
        folder=df_folder,
        config=config,
        folder_config=f"sed_config_momentum_correction{len(features)}.yaml",
        user_config={},
        system_config={},
    )
    processor.generate_splinewarp()
    assert len(processor.mc.pouter_ord) == rotsym
    np.testing.assert_allclose(processor.mc.pouter_ord, pouter_ord)
    np.testing.assert_allclose(processor.mc.cdeform_field, cdeform_field)
    np.testing.assert_allclose(processor.mc.rdeform_field, rdeform_field)
    os.remove(f"sed_config_momentum_correction{len(features)}.yaml")


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
        folder_config={},
        user_config={},
        system_config={},
    )
    with pytest.raises(ValueError):
        processor.apply_momentum_correction()
    processor.bin_and_load_momentum_calibration(apply=True)
    processor.define_features(
        features=feature7,
        rotation_symmetry=6,
        include_center=True,
        apply=True,
    )
    processor.generate_splinewarp(use_center=True)
    processor.pose_adjustment(**adjust_params, apply=True)
    processor.apply_momentum_correction()
    assert "Xm" in processor.dataframe.columns
    assert "Ym" in processor.dataframe.columns


point_a = [308, 345]
k_distance = 4 / 3 * np.pi / 3.28
k_coord_a = [k_distance * 0.3, k_distance * 0.8]


def test_momentum_calibration_workflow():
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
        processor.apply_momentum_calibration()
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
    processor.save_momentum_calibration(filename="sed_config_momentum_calibration.yaml")
    processor = SedProcessor(
        folder=df_folder,
        config=config,
        folder_config="sed_config_momentum_calibration.yaml",
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
    os.remove("sed_config_momentum_calibration.yaml")


def test_energy_correction():
    """Test energy correction workflow."""
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
        processor.apply_energy_correction()
    with pytest.raises(ValueError):
        processor.adjust_energy_correction(apply=True)
    assert processor._pre_binned is not None  # pylint: disable=protected-access
    processor.adjust_energy_correction(
        correction_type="Lorentzian",
        amplitude=2.5,
        center=(730, 730),
        gamma=920,
        apply=True,
    )
    assert processor.ec.correction["correction_type"] == "Lorentzian"
    processor.save_energy_correction()
    processor = SedProcessor(
        folder=df_folder,
        config=config,
        user_config={},
        system_config={},
    )
    processor.adjust_energy_correction(apply=True)
    assert processor.ec.correction["correction_type"] == "Lorentzian"
    processor = SedProcessor(
        folder=df_folder,
        config=config,
        user_config={},
        system_config={},
    )
    processor.apply_energy_correction()
    assert "tm" in processor.dataframe.columns
    os.remove("sed_config.yaml")


energy_scales = ["kinetic", "binding"]
calibration_methods = ["lmfit", "lstsq", "lsqr"]


@pytest.mark.parametrize(
    "energy_scale, calibration_method",
    itertools.product(energy_scales, calibration_methods),
)
def test_energy_calibration_workflow(energy_scale: str, calibration_method: str):
    """Test energy calibration workflow

    Args:
        energy_scale (str): Energy scale
        calibration_method (str): _description_
    """
    config = parse_config(
        config={"core": {"loader": "mpes"}},
        folder_config={},
        user_config={},
        system_config={},
    )
    processor = SedProcessor(
        folder=df_folder + "../mpes/",
        config=config,
        folder_config={},
        user_config={},
        system_config={},
    )
    with pytest.raises(ValueError):
        processor.load_bias_series(data_files=glob.glob(df_folder + "../mpes/*.h5"), normalize=True)
    processor.load_bias_series(
        data_files=glob.glob(df_folder + "../mpes/*.h5"),
        normalize=True,
        bias_key="@KTOF:Lens:Sample:V",
    )
    assert len(processor.ec.biases) == 2
    # load test data into class
    processor.ec.load_data(biases=biases, traces=traces, tof=tof)
    processor.ec.normalize()
    ref_id = 5
    rng = (66100, 67000)
    processor.find_bias_peaks(ranges=rng, ref_id=ref_id, infer_others=True, apply=True)
    ranges: List[Tuple[Any, ...]] = [
        (64638.0, 65386.0),
        (64913.0, 65683.0),
        (65188.0, 65991.0),
        (65474.0, 66310.0),
        (65782.0, 66651.0),
        (66101.0, 67003.0),
        (66442.0, 67388.0),
        (66794.0, 67795.0),
        (67190.0, 68213.0),
        (67575.0, 68664.0),
        (67993.0, 69148.0),
    ]
    processor.find_bias_peaks(ranges=ranges, infer_others=False, apply=True)
    np.testing.assert_allclose(processor.ec.featranges, ranges)
    ref_id = 4
    ref_energy = -0.5
    with pytest.raises(ValueError):
        processor.calibrate_energy_axis(
            ref_energy=ref_energy,
            ref_id=ref_id,
            energy_scale="myfantasyscale",
        )
    with pytest.raises(NotImplementedError):
        processor.calibrate_energy_axis(
            ref_energy=ref_energy,
            ref_id=ref_id,
            method="myfantasymethod",
        )
    processor.calibrate_energy_axis(
        ref_energy=ref_energy,
        ref_id=ref_id,
        energy_scale=energy_scale,
        method=calibration_method,
    )
    assert processor.ec.calibration["energy_scale"] == energy_scale
    processor.save_energy_calibration(
        filename=f"sed_config_energy_calibration_{energy_scale}-{calibration_method}.yaml",
    )
    processor.append_energy_axis()
    assert "energy" in processor.dataframe.columns
    processor = SedProcessor(
        folder=df_folder + "../mpes/",
        config=config,
        folder_config=f"sed_config_energy_calibration_{energy_scale}-{calibration_method}.yaml",
        user_config={},
        system_config={},
    )
    processor.append_energy_axis(preview=True)
    assert "energy" in processor.dataframe.columns
    assert processor.attributes["energy_calibration"]["calibration"]["energy_scale"] == energy_scale
    os.remove(f"sed_config_energy_calibration_{energy_scale}-{calibration_method}.yaml")


def test_delay_calibration_workflow():
    """Test the delay calibration workflow"""
    config = parse_config(
        config={"core": {"loader": "mpes"}},
        folder_config={},
        user_config={},
        system_config={},
    )
    processor = SedProcessor(
        folder=df_folder + "../mpes/",
        config=config,
        folder_config={},
        user_config={},
        system_config={},
    )
    delay_range = (-500, 1500)
    processor.calibrate_delay_axis(delay_range=delay_range, preview=False)
    # read from datafile
    with pytest.raises(NotImplementedError):
        processor.calibrate_delay_axis(preview=True)
    processor.calibrate_delay_axis(
        p1_key="@trARPES:DelayStage:p1",
        p2_key="@trARPES:DelayStage:p2",
        t0_key="@trARPES:DelayStage:t0",
        preview=True,
    )
    assert "delay" in processor.dataframe.columns


def test_add_jitter():
    """Test the jittering function"""
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
    res1 = processor.dataframe["X"].compute()
    processor.add_jitter()
    res2 = processor.dataframe["X"].compute()
    np.testing.assert_allclose(res1, np.round(res1))
    np.testing.assert_allclose(res1, np.round(res2))
    assert (res1 != res2).all()


def test_event_histogram():
    """Test histogram plotting function"""
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
    processor.view_event_histogram(dfpid=0)
    with pytest.raises(ValueError):
        processor.view_event_histogram(dfpid=5)


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


metadata: Dict[Any, Any] = {}
metadata["entry_title"] = "Title"
# User
metadata["user0"] = {}
metadata["user0"]["name"] = "Name"
metadata["user0"]["email"] = "email"
# NXinstrument
metadata["instrument"] = {}
metadata["instrument"] = {}
# analyzer
metadata["instrument"]["analyzer"] = {}
metadata["instrument"]["analyzer"]["energy_resolution"] = 110.0
metadata["instrument"]["analyzer"]["momentum_resolution"] = 0.08
# probe beam
metadata["instrument"]["beam"] = {}
metadata["instrument"]["beam"]["probe"] = {}
metadata["instrument"]["beam"]["probe"]["incident_energy"] = 21.7
# sample
metadata["sample"] = {}
metadata["sample"]["preparation_date"] = "2019-01-13T10:00:00+00:00"
metadata["sample"]["name"] = "Sample Name"


def test_save():
    """Test the save functionality"""
    config = parse_config(
        config={"dataframe": {"tof_binning": 1}},
        folder_config={},
        user_config=package_dir + "/../sed/config/mpes_example_config.yaml",
        system_config={},
    )
    processor = SedProcessor(
        folder=df_folder + "../mpes/",
        config=config,
        folder_config={},
        user_config={},
        system_config={},
        metadata=metadata,
        collect_metadata=True,
    )
    processor.apply_momentum_calibration()
    processor.append_energy_axis()
    processor.calibrate_delay_axis()
    with pytest.raises(NameError):
        processor.save("output.tiff")
    axes = ["kx", "ky", "energy", "delay"]
    bins = [100, 100, 200, 50]
    ranges = [[-2, 2], [-2, 2], [-4, 2], [-600, 1600]]
    processor.compute(bins=bins, axes=axes, ranges=ranges)
    with pytest.raises(NotImplementedError):
        processor.save("output.jpeg")
    processor.save("output.tiff")
    assert os.path.isfile("output.tiff")
    os.remove("output.tiff")
    processor.save("output.h5")
    assert os.path.isfile("output.h5")
    os.remove("output.h5")
    processor.save(
        "output.nxs",
        input_files=df_folder + "../../../../sed/config/NXmpes_config.json",
    )
    assert os.path.isfile("output.nxs")
    os.remove("output.nxs")

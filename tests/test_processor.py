"""Module tests.processor, tests for the sed.core.processor module
"""
from __future__ import annotations

import csv
import glob
import itertools
import json
import logging
import os
import tempfile
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import dask.dataframe as ddf
import numpy as np
import pytest
import xarray as xr
import yaml
from pynxtools.dataconverter.convert import ValidationFailed

from sed import SedProcessor
from sed.core.config import parse_config
from sed.loader.loader_interface import get_loader

package_dir = os.path.dirname(find_spec("sed").origin)
test_dir = os.path.dirname(__file__)
df_folder = f"{test_dir}/data/loader/mpes/"
df_folder_generic = f"{test_dir}/data/loader/generic/"
calibration_folder = f"{test_dir}/data/calibrator/"
files = glob.glob(df_folder + "*.h5")
runs = ["30", "50"]
runs_flash = ["43878", "43878"]
loader = get_loader(loader_name="mpes")
source_folder = f"{test_dir}/../"
dest_folder = tempfile.mkdtemp()
gid = os.getgid()

traces_list = []
with open(calibration_folder + "traces.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        traces_list.append(row)
traces = np.asarray(traces_list).T
with open(calibration_folder + "tof.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    tof = np.asarray(next(reader))
with open(calibration_folder + "biases.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    biases = np.asarray(next(reader))


def test_processor_from_dataframe() -> None:
    """Test generation of the processor from a dataframe object"""
    config = {"core": {"loader": "mpes"}}
    dataframe, timed_dataframe, _ = loader.read_dataframe(files=files)
    processor = SedProcessor(
        dataframe=dataframe,
        timed_dataframe=timed_dataframe,
        config=config,
        folder_config={},
        user_config={},
        system_config={},
        verbose=True,
    )
    for column in dataframe.columns:
        assert (dataframe[column].compute() == processor.dataframe[column].compute()).all()
    for column in timed_dataframe.columns:
        assert (
            timed_dataframe[column].compute() == processor.timed_dataframe[column].compute()
        ).all()


def test_processor_from_files() -> None:
    """Test generation of the processor from a list of files"""
    config = {"core": {"loader": "mpes"}}
    dataframe, timed_dataframe, _ = loader.read_dataframe(files=files)
    processor = SedProcessor(
        files=files,
        config=config,
        folder_config={},
        user_config={},
        system_config={},
        verbose=True,
    )
    for column in dataframe.columns:
        assert (dataframe[column].compute() == processor.dataframe[column].compute()).all()
    for column in timed_dataframe.columns:
        assert (
            timed_dataframe[column].compute() == processor.timed_dataframe[column].compute()
        ).all()


def test_processor_from_folders() -> None:
    """Test generation of the processor from a folder"""
    config = {"core": {"loader": "mpes"}}
    dataframe, timed_dataframe, _ = loader.read_dataframe(folders=df_folder)
    processor = SedProcessor(
        folder=df_folder,
        config=config,
        folder_config={},
        user_config={},
        system_config={},
        verbose=True,
    )
    for column in dataframe.columns:
        assert (dataframe[column].compute() == processor.dataframe[column].compute()).all()
    for column in timed_dataframe.columns:
        assert (
            timed_dataframe[column].compute() == processor.timed_dataframe[column].compute()
        ).all()


def test_processor_from_runs() -> None:
    """Test generation of the processor from runs"""
    config = {"core": {"loader": "mpes"}}
    dataframe, timed_dataframe, _ = loader.read_dataframe(folders=df_folder, runs=runs)
    processor = SedProcessor(
        folder=df_folder,
        config=config,
        runs=runs,
        folder_config={},
        user_config={},
        system_config={},
        verbose=True,
    )
    assert processor.loader.runs == runs
    for column in dataframe.columns:
        assert (dataframe[column].compute() == processor.dataframe[column].compute()).all()
    for column in timed_dataframe.columns:
        assert (
            timed_dataframe[column].compute() == processor.timed_dataframe[column].compute()
        ).all()


def test_additional_parameter_to_loader() -> None:
    """Test if additional keyword parameter can be passed to the loader from the
    Processor initialization.
    """
    config = {"core": {"loader": "generic"}}
    processor = SedProcessor(
        folder=df_folder_generic,
        ftype="json",
        config=config,
        folder_config={},
        user_config={},
        system_config={},
        verbose=True,
    )
    assert processor.files[0].find("json") > -1

    # check that illegal keywords raise:
    with pytest.raises(TypeError):
        processor = SedProcessor(
            folder=df_folder_generic,
            ftype="json",
            config=config,
            folder_config={},
            user_config={},
            system_config={},
            verbose=True,
            illegal_keyword=True,
        )


def test_repr() -> None:
    """test the ___repr___ method"""
    config = {"core": {"loader": "mpes"}}
    processor = SedProcessor(
        config=config,
        folder_config={},
        user_config={},
        system_config={},
        verbose=True,
    )
    processor_str = str(processor)
    assert processor_str.find("No Data loaded") > 0
    with pytest.raises(ValueError):
        processor.load()
    processor.load(files=files, metadata={"test": {"key1": "value1"}})
    processor_str = str(processor)
    assert processor_str.find("ADC") > 0
    assert processor_str.find("key1") > 0

    with pytest.raises(TypeError):
        processor.load(files=files, metadata={"test": {"key1": "value1"}}, illegal_keyword=True)


def test_attributes_setters() -> None:
    """Test class attributes and setters."""
    config = {"core": {"loader": "mpes"}}
    processor = SedProcessor(
        config=config,
        folder_config={},
        user_config={},
        system_config={},
        verbose=True,
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
    processor_metadata = processor.attributes.metadata
    assert isinstance(processor_metadata, dict)
    assert "test" in processor_metadata.keys()
    processor.add_attribute({"key2": 5}, name="test2")
    assert processor_metadata["test2"]["key2"] == 5
    assert processor.config["core"]["loader"] == "mpes"
    assert len(processor.files) == 2


def test_copy_tool() -> None:
    """Test the copy tool functionality in the processor"""
    config: dict[str, dict[str, Any]] = {"core": {"loader": "mpes"}}
    processor = SedProcessor(
        config=config,
        folder_config={},
        user_config={},
        system_config={},
        verbose=True,
    )
    assert processor.use_copy_tool is False
    config = {
        "core": {
            "loader": "mpes",
            "copy_tool": {"source": source_folder, "dest": dest_folder, "gid": os.getgid()},
        },
    }
    processor = SedProcessor(
        config=config,
        folder_config={},
        user_config={},
        system_config={},
        verbose=True,
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

adjust_params: dict[str, Any] = {
    "scale": np.random.randint(1, 10) / 10 + 0.5,
    "xtrans": np.random.randint(1, 50),
    "ytrans": np.random.randint(1, 50),
    "angle": np.random.randint(1, 50),
}


@pytest.mark.parametrize(
    "features",
    feature_list,
)
def test_momentum_correction_workflow(features: np.ndarray) -> None:
    """Test for the momentum correction workflow"""
    config = {"core": {"loader": "mpes"}}
    processor = SedProcessor(
        folder=df_folder,
        config=config,
        folder_config={},
        user_config={},
        system_config={},
        verbose=True,
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
        verbose=True,
    )
    processor.generate_splinewarp()
    assert len(processor.mc.pouter_ord) == rotsym
    np.testing.assert_allclose(processor.mc.pouter_ord, pouter_ord)
    np.testing.assert_allclose(processor.mc.cdeform_field, cdeform_field)
    np.testing.assert_allclose(processor.mc.rdeform_field, rdeform_field)
    os.remove(f"sed_config_momentum_correction{len(features)}.yaml")


def test_pose_adjustment() -> None:
    """Test for the pose correction and application of momentum correction workflow"""
    config = {"core": {"loader": "mpes"}}
    processor = SedProcessor(
        folder=df_folder,
        config=config,
        folder_config={},
        user_config={},
        system_config={},
        verbose=True,
    )
    # pose adjustment w/o loaded image
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
        verbose=True,
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

    # illegal keywords:
    with pytest.raises(TypeError):
        processor.pose_adjustment(**adjust_params, apply=True, illegal_kwd=True)


def test_pose_adjustment_save_load() -> None:
    """Test for the saving and loading of pose correction and application of momentum correction
    workflow"""
    config = {"core": {"loader": "mpes"}}
    # pose adjustment w/ loaded image
    processor = SedProcessor(
        folder=df_folder,
        config=config,
        folder_config={},
        user_config={},
        system_config={},
        verbose=True,
    )
    processor.bin_and_load_momentum_calibration(apply=True)
    processor.define_features(
        features=feature7,
        rotation_symmetry=6,
        include_center=True,
        apply=True,
    )
    processor.generate_splinewarp(use_center=True)
    processor.save_splinewarp(filename="sed_config_pose_adjustments.yaml")
    processor.pose_adjustment(**adjust_params, apply=True)  # type: ignore[arg-type]
    processor.save_transformations(filename="sed_config_pose_adjustments.yaml")
    processor.apply_momentum_correction()
    cdeform_field = processor.mc.cdeform_field.copy()
    rdeform_field = processor.mc.rdeform_field.copy()
    # pose adjustment w/o loaded image
    processor = SedProcessor(
        folder=df_folder,
        config=config,
        folder_config="sed_config_pose_adjustments.yaml",
        user_config={},
        system_config={},
        verbose=True,
    )
    processor.pose_adjustment(**adjust_params, apply=True)  # type: ignore[arg-type]
    processor.apply_momentum_correction()
    # pose adjustment w/o loaded image and w/o correction
    processor = SedProcessor(
        folder=df_folder,
        config=config,
        folder_config="sed_config_pose_adjustments.yaml",
        user_config={},
        system_config={},
        verbose=True,
    )
    processor.pose_adjustment(**adjust_params, use_correction=False, apply=True)  # type: ignore[arg-type]
    assert np.all(processor.mc.cdeform_field != cdeform_field)
    assert np.all(processor.mc.rdeform_field != rdeform_field)
    # from config
    processor = SedProcessor(
        folder=df_folder,
        config=config,
        folder_config="sed_config_pose_adjustments.yaml",
        user_config={},
        system_config={},
        verbose=True,
    )
    processor.apply_momentum_correction()
    assert "Xm" in processor.dataframe.columns
    assert "Ym" in processor.dataframe.columns
    assert "momentum_correction" in processor.attributes.metadata
    np.testing.assert_allclose(processor.mc.cdeform_field, cdeform_field)
    np.testing.assert_allclose(processor.mc.rdeform_field, rdeform_field)
    os.remove("sed_config_pose_adjustments.yaml")


point_a = [308, 345]
k_distance = 4 / 3 * np.pi / 3.28
k_coord_a = [k_distance * 0.3, k_distance * 0.8]


def test_momentum_calibration_workflow() -> None:
    """Test the calibration of the momentum axes"""
    config = {"core": {"loader": "mpes"}}
    processor = SedProcessor(
        folder=df_folder,
        config=config,
        folder_config={},
        user_config={},
        system_config={},
        verbose=True,
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
        verbose=True,
    )
    processor.apply_momentum_calibration()
    assert (
        processor.config["momentum"]["calibration"]["kx_scale"]
        != processor.config["momentum"]["calibration"]["ky_scale"]
    )
    assert "kx" in processor.dataframe.columns
    assert "ky" in processor.dataframe.columns
    os.remove("sed_config_momentum_calibration.yaml")


def test_energy_correction() -> None:
    """Test energy correction workflow."""
    config = {"core": {"loader": "mpes"}}
    processor = SedProcessor(
        folder=df_folder,
        config=config,
        folder_config={},
        user_config={},
        system_config={},
        verbose=True,
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
        verbose=True,
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
def test_energy_calibration_workflow(energy_scale: str, calibration_method: str) -> None:
    """Test energy calibration workflow

    Args:
        energy_scale (str): Energy scale
        calibration_method (str): calibration method to use
    """
    config = {"core": {"loader": "mpes"}}
    processor = SedProcessor(
        folder=df_folder,
        config=config,
        folder_config={},
        user_config={},
        system_config={},
        verbose=True,
    )
    with pytest.raises(ValueError):
        processor.load_bias_series()
    with pytest.raises(ValueError):
        processor.load_bias_series(data_files=files, normalize=True)
    processor.load_bias_series(
        data_files=files,
        normalize=True,
        bias_key="@KTOF:Lens:Sample:V",
    )
    assert len(processor.ec.biases) == 2
    # load data as tuple
    with pytest.raises(ValueError):
        processor.load_bias_series(binned_data=(tof, traces))  # type: ignore
    processor.load_bias_series(binned_data=(tof, biases, traces))
    assert processor.ec.traces.shape == traces.shape
    assert len(processor.ec.biases) == processor.ec.traces.shape[0]
    assert len(processor.ec.tof) == processor.ec.traces.shape[1]
    # load data as xarray
    with pytest.raises(ValueError):
        bias_series = xr.DataArray(data=traces, coords={"biases": biases, "tof": tof})
        processor.load_bias_series(binned_data=bias_series)
    bias_series = xr.DataArray(data=traces, coords={"sampleBias": biases, "t": tof})
    processor.load_bias_series(binned_data=bias_series)
    assert processor.ec.traces.shape == traces.shape
    assert len(processor.ec.biases) == processor.ec.traces.shape[0]
    assert len(processor.ec.tof) == processor.ec.traces.shape[1]
    processor.ec.normalize()
    ref_id = 5
    rng = (66100, 67000)
    processor.find_bias_peaks(ranges=rng, ref_id=ref_id, infer_others=True, apply=True)
    ranges: list[tuple[Any, ...]] = [
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
            energy_scale="myfantasyscale",
        )
    with pytest.raises(NotImplementedError):
        processor.calibrate_energy_axis(
            ref_energy=ref_energy,
            method="myfantasymethod",
        )
    processor.calibrate_energy_axis(
        ref_energy=ref_energy,
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
        verbose=True,
    )
    with pytest.raises(ValueError):
        processor.add_energy_offset(constant=1)
    processor.append_energy_axis(preview=False)
    assert "energy" in processor.dataframe.columns
    assert (
        processor.attributes.metadata["energy_calibration"]["calibration"]["energy_scale"]
        == energy_scale
    )
    os.remove(f"sed_config_energy_calibration_{energy_scale}-{calibration_method}.yaml")

    energy1 = processor.dataframe["energy"].compute().values
    processor.add_energy_offset(constant=1)
    energy2 = processor.dataframe["energy"].compute().values
    np.testing.assert_allclose(energy1, energy2 + (1 if energy_scale == "binding" else -1))


def test_align_dld_sectors() -> None:
    """Test alignment of DLD sectors for flash detector"""
    config = parse_config(
        df_folder + "../flash/config.yaml",
        folder_config={},
        user_config={},
        system_config={},
    )
    config["core"]["paths"]["processed"] = Path(
        config["core"]["paths"]["processed"],
        "_align_dld_sectors",
    )
    processor = SedProcessor(
        folder=df_folder + "../flash/",
        config=config,
        runs=runs_flash,
        folder_config={},
        user_config={},
        system_config={},
        verbose=True,
    )
    assert "dldTimeSteps" in processor.dataframe.columns
    assert "dldSectorID" in processor.dataframe.columns

    sector_delays = np.asarray([10, -10, 20, -20, 30, -30, 40, -40])
    tof_ref = []
    for i in range(len(sector_delays)):
        tof_ref.append(
            processor.dataframe[processor.dataframe["dldSectorID"] == i]["dldTimeSteps"]
            .compute()
            .values.astype("float"),
        )
    tof_ref_array = np.zeros([len(tof_ref), len(max(tof_ref, key=len))])
    tof_ref_array[:] = np.nan
    for i, val in enumerate(tof_ref):
        tof_ref_array[i][0 : len(val)] = val
    processor.align_dld_sectors(sector_delays=sector_delays)
    tof_aligned = []
    for i in range(len(sector_delays)):
        tof_aligned.append(
            processor.dataframe[processor.dataframe["dldSectorID"] == i]["dldTimeSteps"]
            .compute()
            .values,
        )
    tof_aligned_array = np.zeros([len(tof_aligned), len(max(tof_aligned, key=len))])
    tof_aligned_array[:] = np.nan
    for i, val in enumerate(tof_aligned):
        tof_aligned_array[i][0 : len(val)] = val
    np.testing.assert_allclose(tof_ref_array, tof_aligned_array + sector_delays[:, np.newaxis])

    # cleanup flash intermediaries
    parquet_data_dir = config["core"]["paths"]["processed"]
    for file in os.listdir(Path(parquet_data_dir, "buffer")):
        os.remove(Path(parquet_data_dir, "buffer", file))


def test_append_tof_ns_axis() -> None:
    """Test the append_tof_ns_axis function"""
    config = {"core": {"loader": "mpes"}}
    processor = SedProcessor(
        folder=df_folder,
        config=config,
        folder_config={},
        user_config={},
        system_config={},
        verbose=True,
    )
    processor.append_tof_ns_axis()
    assert processor.config["dataframe"]["columns"]["tof_ns"] in processor.dataframe


def test_delay_calibration_workflow() -> None:
    """Test the delay calibration workflow"""
    config = {"core": {"loader": "mpes"}}
    processor = SedProcessor(
        folder=df_folder,
        config=config,
        folder_config={},
        user_config={},
        system_config={},
        verbose=True,
    )
    delay_range = (-500, 1500)
    processor.calibrate_delay_axis(delay_range=delay_range)
    # read from datafile
    processor = SedProcessor(
        folder=df_folder,
        config=config,
        folder_config={},
        user_config={},
        system_config={},
        verbose=True,
    )
    with pytest.raises(ValueError):
        processor.add_delay_offset(constant=1)
    with pytest.raises(NotImplementedError):
        processor.calibrate_delay_axis()
    with pytest.raises(NotImplementedError):
        processor.calibrate_delay_axis(
            p1_key="@trARPES:DelayStage:p1",
            p2_key="@trARPES:DelayStage:p2",
            t0_key="@trARPES:DelayStage:t0",
            read_delay_ranges=False,
        )
    processor.calibrate_delay_axis(
        p1_key="@trARPES:DelayStage:p1",
        p2_key="@trARPES:DelayStage:p2",
        t0_key="@trARPES:DelayStage:t0",
        read_delay_ranges=True,
    )
    assert "delay" in processor.dataframe.columns
    creation_date_calibration = processor.dc.calibration["creation_date"]
    expected = -1 * (
        processor.dataframe["delay"].compute() + 1 + processor.dataframe["ADC"].compute()
    )
    processor.add_delay_offset(constant=1, columns="ADC", flip_delay_axis=True)
    creation_date_offsets = processor.dc.offsets["creation_date"]
    np.testing.assert_allclose(expected, processor.dataframe["delay"].compute())
    # test saving and loading
    processor.save_delay_calibration(filename="sed_config_delay_calibration.yaml")
    processor.save_delay_offsets(filename="sed_config_delay_calibration.yaml")
    processor = SedProcessor(
        folder=df_folder + "../mpes/",
        config=config,
        folder_config="sed_config_delay_calibration.yaml",
        user_config={},
        system_config={},
        verbose=True,
    )
    processor.calibrate_delay_axis(read_delay_ranges=False)
    assert "delay" in processor.dataframe.columns
    assert (
        processor.attributes.metadata["delay_calibration"]["calibration"]["creation_date"]
        == creation_date_calibration
    )
    processor.add_delay_offset(preview=True)
    assert (
        processor.attributes.metadata["delay_offset"]["offsets"]["creation_date"]
        == creation_date_offsets
    )
    np.testing.assert_allclose(expected, processor.dataframe["delay"].compute())
    os.remove("sed_config_delay_calibration.yaml")


def test_filter_column() -> None:
    """Test the jittering function"""
    config = {"core": {"loader": "mpes"}}
    processor = SedProcessor(
        folder=df_folder,
        config=config,
        folder_config={},
        user_config={},
        system_config={},
        verbose=True,
    )
    low, high = np.quantile(processor.dataframe["X"].compute(), [0.1, 0.9])
    processor.filter_column("X", low, high)
    assert processor.dataframe["X"].compute().min() >= low
    assert processor.dataframe["X"].compute().max() <= high
    with pytest.raises(KeyError):
        processor.filter_column("wrong", low, high)
    with pytest.raises(ValueError):
        processor.filter_column("X", high, low)


def test_add_jitter() -> None:
    """Test the jittering function"""
    config = {"core": {"loader": "mpes"}}
    processor = SedProcessor(
        folder=df_folder,
        config=config,
        folder_config={},
        user_config={},
        system_config={},
        verbose=True,
    )
    res1 = processor.dataframe["X"].compute()
    res1a = processor.dataframe["ADC"].compute()
    processor.add_jitter()
    res2 = processor.dataframe["X"].compute()
    res2a = processor.dataframe["ADC"].compute()
    np.testing.assert_allclose(res1, np.round(res1))
    np.testing.assert_allclose(res1, np.round(res2))
    assert (res1 != res2).all()
    # test that jittering is not applied on ADC column
    np.testing.assert_allclose(res1a, res2a)


def test_add_time_stamped_data() -> None:
    """Test the function to add time-stamped data"""
    processor = SedProcessor(
        folder=df_folder + "../mpes/",
        config=package_dir + "/config/mpes_example_config.yaml",
        folder_config={},
        user_config={},
        system_config={},
        time_stamps=True,
        verbose=True,
        verify_config=False,
    )
    df_ts = processor.dataframe.timeStamps.compute().values
    data = np.linspace(0, 1, 20)
    time_stamps = np.linspace(df_ts[0], df_ts[-1], 20)
    processor.add_time_stamped_data(
        time_stamps=time_stamps,
        data=data,
        dest_column="time_stamped_data",
    )
    assert "time_stamped_data" in processor.dataframe
    res = processor.dataframe["time_stamped_data"].compute().values
    assert res[0] == 0
    assert res[-1] == 1
    assert processor.attributes.metadata["time_stamped_data"][0] == "time_stamped_data"
    np.testing.assert_array_equal(
        processor.attributes.metadata["time_stamped_data"][1],
        time_stamps,
    )
    np.testing.assert_array_equal(processor.attributes.metadata["time_stamped_data"][2], data)


def test_event_histogram() -> None:
    """Test histogram plotting function"""
    config = {"core": {"loader": "mpes"}}
    processor = SedProcessor(
        folder=df_folder,
        config=config,
        folder_config={},
        user_config={},
        system_config={},
        verbose=True,
    )
    processor.view_event_histogram(dfpid=0)
    with pytest.raises(ValueError):
        processor.view_event_histogram(dfpid=5)


def test_compute() -> None:
    """Test binning of final result"""
    config = {"core": {"loader": "mpes"}}
    processor = SedProcessor(
        folder=df_folder,
        config=config,
        folder_config={},
        user_config={},
        system_config={},
        verbose=True,
    )
    bins = [10, 10, 10, 10]
    axes = ["X", "Y", "t", "ADC"]
    ranges = [(0, 2048), (0, 2048), (0, 200000), (0, 50000)]
    result = processor.compute(bins=bins, axes=axes, ranges=ranges, df_partitions=5)
    assert result.data.shape == tuple(bins)
    assert result.data.sum(axis=(0, 1, 2, 3)) > 0

    # illegal keywords:
    with pytest.raises(TypeError):
        processor.compute(illegal_kwd=True)


def test_compute_with_filter() -> None:
    """Test binning of final result using filters"""
    config = {"core": {"loader": "mpes"}}
    processor = SedProcessor(
        folder=df_folder,
        config=config,
        folder_config={},
        user_config={},
        system_config={},
        verbose=True,
    )
    bins = [10, 10, 10, 10]
    axes = ["X", "Y", "t", "ADC"]
    ranges = [(0, 2048), (0, 2048), (0, 200000), (0, 50000)]
    filters = [
        {"col": "X", "lower_bound": 100, "upper_bound": 200},
        {"col": "index", "lower_bound": 100, "upper_bound": 200},
    ]
    result = processor.compute(bins=bins, axes=axes, ranges=ranges, df_partitions=5)
    result_filtered = processor.compute(
        bins=bins,
        axes=axes,
        ranges=ranges,
        df_partitions=5,
        filter=filters,
    )
    assert result.sum(axis=(0, 1, 2, 3)) > result_filtered.sum(axis=(0, 1, 2, 3))
    with pytest.raises(ValueError) as e:
        processor.compute(
            bins=bins,
            axes=axes,
            ranges=ranges,
            df_partitions=5,
            filter=[{"lower_bound": 100}],
        )
    assert str(e.value.args[0]).find("'col' needs to be defined for each filter entry!") > -1

    with pytest.raises(ValueError) as e:
        processor.compute(
            bins=bins,
            axes=axes,
            ranges=ranges,
            df_partitions=5,
            filter=[{"col": "X", "invalid": 100}],
        )
    assert (
        str(e.value.args[0]).find(
            "Only 'col', 'lower_bound' and 'upper_bound' allowed as filter entries. ",
        )
        > -1
    )


def test_compute_with_normalization() -> None:
    """Test binning of final result with histogram normalization"""
    config = {"core": {"loader": "mpes"}}
    processor = SedProcessor(
        folder=df_folder,
        config=config,
        folder_config={},
        user_config={},
        system_config={},
        verbose=True,
    )
    bins = [10, 10, 10, 5]
    axes = ["X", "Y", "t", "ADC"]
    ranges = [(0, 2048), (0, 2048), (0, 200000), (650, 655)]
    result = processor.compute(
        bins=bins,
        axes=axes,
        ranges=ranges,
        df_partitions=5,
        normalize_to_acquisition_time="ADC",
    )
    assert result.data.shape == tuple(bins)
    assert result.data.sum(axis=(0, 1, 2, 3)) > 0
    assert processor.normalization_histogram is not None
    assert processor.normalized is not None
    np.testing.assert_allclose(
        processor.binned.data,
        (processor.normalized * processor.normalization_histogram).data,
    )
    # bin only second dataframe partition
    result2 = processor.compute(
        bins=bins,
        axes=axes,
        ranges=ranges,
        df_partitions=[1],
        normalize_to_acquisition_time="ADC",
    )
    # Test that the results normalize roughly to the same count rate
    assert abs(result.sum(axis=(0, 1, 2, 3)) / result2.sum(axis=(0, 1, 2, 3)) - 1) < 0.15


def test_get_normalization_histogram() -> None:
    """Test the generation function for the normalization histogram"""
    config = {"core": {"loader": "mpes"}, "dataframe": {"columns": {"timestamp": "timeStamps"}}}
    processor = SedProcessor(
        folder=df_folder,
        config=config,
        folder_config={},
        user_config={},
        system_config={},
        time_stamps=True,
        verbose=True,
    )
    bins = [10, 10, 10, 5]
    axes = ["X", "Y", "t", "ADC"]
    ranges = [(0, 2048), (0, 2048), (0, 200000), (650, 655)]
    with pytest.raises(ValueError):
        processor.get_normalization_histogram(axis="ADC")
    processor.compute(bins=bins, axes=axes, ranges=ranges, df_partitions=5)
    with pytest.raises(ValueError):
        processor.get_normalization_histogram(axis="Delay")
    histogram1 = processor.get_normalization_histogram(axis="ADC", df_partitions=1)
    histogram2 = processor.get_normalization_histogram(
        axis="ADC",
        use_time_stamps=True,
        df_partitions=1,
    )
    # TODO: Check why histograms are so different
    np.testing.assert_allclose(
        histogram1 / histogram1.sum(),
        histogram2 / histogram2.sum(),
        atol=0.02,
    )
    # histogram1 = processor.get_normalization_histogram(axis="ADC")
    # histogram2 = processor.get_normalization_histogram(axis="ADC", use_time_stamps="True")
    # np.testing.assert_allclose(histogram1, histogram2)

    # illegal keywords:
    with pytest.raises(TypeError):
        histogram1 = processor.get_normalization_histogram(axis="ADC", illegal_kwd=True)


metadata: dict[Any, Any] = {}
metadata["entry_title"] = "Title"
# user
metadata["user0"] = {}
metadata["user0"]["name"] = "Name"
metadata["user0"]["email"] = "email"
metadata["user0"]["affiliation"] = "affiliation"
# instrument
metadata["instrument"] = {}
metadata["instrument"]["energy_resolution"] = 150.0
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


def test_save(caplog) -> None:
    """Test the save functionality"""
    config = parse_config(
        config={"dataframe": {"tof_binning": 1}},
        folder_config={},
        user_config=package_dir + "/config/mpes_example_config.yaml",
        system_config={},
        verify_config=False,
    )
    config["metadata"]["lens_mode_config"]["6kV_kmodem4.0_30VTOF_453ns_focus.sav"][
        "MCPfront"
    ] = 21.0
    config["metadata"]["lens_mode_config"]["6kV_kmodem4.0_30VTOF_453ns_focus.sav"]["Z1"] = 2450
    config["metadata"]["lens_mode_config"]["6kV_kmodem4.0_30VTOF_453ns_focus.sav"]["F"] = 69.23
    config["nexus"]["input_files"] = [package_dir + "/config/NXmpes_config.json"]
    processor = SedProcessor(
        folder=df_folder,
        config=config,
        folder_config={},
        user_config={},
        system_config={},
        metadata=metadata,
        collect_metadata=True,
        verbose=True,
    )
    processor.apply_momentum_calibration()
    processor.append_energy_axis()
    processor.calibrate_delay_axis()
    with pytest.raises(NameError):
        processor.save("output.tiff")
    axes = ["kx", "ky", "energy", "delay"]
    bins = [100, 100, 200, 50]
    ranges = [(-2, 2), (-2, 2), (-4, 2), (-600, 1600)]
    processor.compute(bins=bins, axes=axes, ranges=ranges)
    with pytest.raises(NotImplementedError):
        processor.save("output.jpeg")
    processor.save("output.tiff")
    assert os.path.isfile("output.tiff")
    os.remove("output.tiff")
    processor.save("output.h5")
    assert os.path.isfile("output.h5")
    os.remove("output.h5")
    # convert using the fail=True keyword. This ensures that the pynxtools backend will throw
    # and error if any validation problems occur.
    processor.save(
        "output.nxs",
        fail=True,
    )
    assert os.path.isfile("output.nxs")
    # Test that a validation error is raised if a required field is missing
    del processor.binned.attrs["metadata"]["sample"]["name"]
    with pytest.raises(ValidationFailed):
        processor.save(
            "result.nxs",
            fail=True,
        )
    # Check that the issues are raised as warnings per default:
    caplog.clear()
    with open("temp_eln.yaml", "w") as f:
        yaml.dump({"Instrument": {"undocumented_field": "undocumented entry"}}, f)
    with open("temp_config.json", "w") as f:
        with open(
            package_dir + "/config/NXmpes_config.json",
            encoding="utf-8",
        ) as stream:
            config_dict = json.load(stream)
        config_dict[
            "/ENTRY[entry]/INSTRUMENT[instrument]/undocumented_field"
        ] = "@eln:/ENTRY/Instrument/undocumented_field"
        json.dump(config_dict, f, indent=2)
    with caplog.at_level(logging.WARNING):
        processor.save(
            "result.nxs",
            input_files=["temp_config.json"],
            eln_data="temp_eln.yaml",
            suppress_warning=True,
        )
        assert (
            caplog.messages[0]
            == "WARNING: The data entry corresponding to /ENTRY[entry]/SAMPLE[sample]/name is "
            "required and hasn't been supplied by the reader."
        )
        assert (
            caplog.messages[1]
            == "WARNING: Field /ENTRY[entry]/INSTRUMENT[instrument]/undocumented_field written "
            "without documentation."
        )
    os.remove("output.nxs")
    os.remove("temp_eln.yaml")
    os.remove("temp_config.json")

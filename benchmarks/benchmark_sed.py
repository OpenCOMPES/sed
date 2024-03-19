"""This file contains code that performs benchmarks for the processor workflows
"""
import os
import timeit
from importlib.util import find_spec

import dask
import numpy as np
import psutil

from sed import SedProcessor
from sed.binning.binning import bin_dataframe
from sed.core.config import load_config
from sed.core.config import save_config

package_dir = os.path.dirname(find_spec("sed").origin)


num_cores = min(20, psutil.cpu_count())
# use fix random numbers for comparability
np.random.seed(42)
# 100 Billion events, ~ 3 GByte.
n_pts = 100000000
ranges = np.array([[0, 2048], [0, 2048], [60000, 120000], [2000, 20000]])
axes = ["X", "Y", "t", "ADC"]
array = (
    dask.array.random.random((n_pts, len(ranges))) * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]
)
dataframe = dask.dataframe.from_dask_array(array, columns=axes)


targets = load_config(package_dir + "/../benchmarks/benchmark_targets.yaml")


def test_binning_1d() -> None:
    """Run a benchmark for 1d binning of artificial data"""
    bins_ = [1000]
    axes_ = ["t"]
    ranges_ = [(60000, 120000)]
    bin_dataframe(df=dataframe.copy(), bins=bins_, axes=axes_, ranges=ranges_, n_cores=num_cores)
    command = (
        "bin_dataframe(df=dataframe.copy(), bins=bins_, axes=axes_, "
        "ranges=ranges_, n_cores=num_cores)"
    )
    timer = timeit.Timer(
        command,
        globals={**globals(), **locals()},
    )
    result = timer.repeat(5, number=1)
    print(result)
    assert min(result) < targets["binning_1d"]
    # update targets if substantial improvement occurs
    if np.mean(result) < 0.9 * targets["binning_1d"]:
        targets["binning_1d"] = float(np.mean(result) * 1.1)
        save_config(targets, package_dir + "/../benchmarks/benchmark_targets.yaml")


def test_binning_4d() -> None:
    """Run a benchmark for 4d binning of artificial data"""
    bins_ = [100, 100, 100, 100]
    axes_ = axes
    ranges_ = [(0, 2048), (0, 2048), (60000, 120000), (2000, 20000)]
    bin_dataframe(df=dataframe.copy(), bins=bins_, axes=axes_, ranges=ranges_, n_cores=num_cores)
    command = (
        "bin_dataframe(df=dataframe.copy(), bins=bins_, axes=axes_, "
        "ranges=ranges_, n_cores=num_cores)"
    )
    timer = timeit.Timer(
        command,
        globals={**globals(), **locals()},
    )
    result = timer.repeat(5, number=1)
    print(result)
    assert min(result) < targets["binning_4d"]
    # update targets if substantial improvement occurs
    if np.mean(result) < 0.9 * targets["binning_4d"]:
        targets["binning_4d"] = float(np.mean(result) * 1.1)
        save_config(targets, package_dir + "/../benchmarks/benchmark_targets.yaml")


def test_splinewarp() -> None:
    """Run a benchmark for the generation of the inverse dfield correction"""
    processor = SedProcessor(
        dataframe=dataframe.copy(),
        config=package_dir + "/config/mpes_example_config.yaml",
        folder_config={},
        user_config={},
        system_config={},
        verbose=True,
    )
    processor.apply_momentum_correction()
    timer = timeit.Timer(
        "processor.mc.dfield_updated=True; processor.apply_momentum_correction()",
        globals={**globals(), **locals()},
    )
    result = timer.repeat(5, number=1)
    print(result)
    assert min(result) < targets["inv_dfield"]
    # update targets if substantial improvement occurs
    if np.mean(result) < 0.9 * targets["inv_dfield"]:
        targets["inv_dfield"] = float(np.mean(result) * 1.1)
        save_config(targets, package_dir + "/../benchmarks/benchmark_targets.yaml")


def test_workflow_1d() -> None:
    """Run a benchmark for 1d binning of converted data"""
    processor = SedProcessor(
        dataframe=dataframe.copy(),
        config=package_dir + "/config/mpes_example_config.yaml",
        folder_config={},
        user_config={},
        system_config={},
        verbose=True,
    )
    processor.add_jitter()
    processor.apply_momentum_correction()
    processor.apply_momentum_calibration()
    processor.apply_energy_correction()
    processor.append_energy_axis()
    processor.calibrate_delay_axis(delay_range=(-500, 1500))
    bins_ = [1000]
    axes_ = ["energy"]
    ranges_ = [(-10, 10)]
    processor.compute(bins=bins_, axes=axes_, ranges=ranges_)
    timer = timeit.Timer(
        "processor.compute(bins=bins_, axes=axes_, ranges=ranges_)",
        globals={**globals(), **locals()},
    )
    result = timer.repeat(5, number=1)
    print(result)
    assert min(result) < targets["workflow_1d"]
    # update targets if substantial improvement occurs
    if np.mean(result) < 0.9 * targets["workflow_1d"]:
        targets["workflow_1d"] = float(np.mean(result) * 1.1)
        save_config(targets, package_dir + "/../benchmarks/benchmark_targets.yaml")


def test_workflow_4d() -> None:
    """Run a benchmark for 4d binning of converted data"""
    processor = SedProcessor(
        dataframe=dataframe.copy(),
        config=package_dir + "/config/mpes_example_config.yaml",
        folder_config={},
        user_config={},
        system_config={},
        verbose=True,
    )
    processor.add_jitter()
    processor.apply_momentum_correction()
    processor.apply_momentum_calibration()
    processor.apply_energy_correction()
    processor.append_energy_axis()
    processor.calibrate_delay_axis(delay_range=(-500, 1500))
    bins_ = [100, 100, 100, 100]
    axes_ = ["kx", "ky", "energy", "delay"]
    ranges_ = [(-2, 2), (-2, 2), (-10, 10), (-1000, 1000)]
    processor.compute(bins=bins_, axes=axes_, ranges=ranges_)
    timer = timeit.Timer(
        "processor.compute(bins=bins_, axes=axes_, ranges=ranges_)",
        globals={**globals(), **locals()},
    )
    result = timer.repeat(5, number=1)
    print(result)
    assert min(result) < targets["workflow_4d"]
    # update targets if substantial improvement occurs
    if np.mean(result) < 0.9 * targets["workflow_4d"]:
        targets["workflow_4d"] = float(np.mean(result) * 1.1)
        save_config(targets, package_dir + "/../benchmarks/benchmark_targets.yaml")

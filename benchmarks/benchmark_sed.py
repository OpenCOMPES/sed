"""This file contains code that performs benchmarks for the processor workflows
"""
import os
import timeit
from importlib.util import find_spec

import dask
import numpy as np
import psutil
import pytest

from sed import SedProcessor
from sed.binning.binning import bin_dataframe
from sed.core.config import load_config
from sed.core.config import save_config
from sed.loader.base.loader import BaseLoader
from tests.loader.test_loaders import get_all_loaders
from tests.loader.test_loaders import get_loader_name_from_loader_object

package_dir = os.path.dirname(find_spec("sed").origin)
benchmark_dir = os.path.dirname(__file__)


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

test_data_dir = os.path.join(benchmark_dir, "..", "tests", "data")
runs = {"generic": None, "mpes": ["30", "50"], "flash": ["43878"], "sxp": ["0016"]}

targets = load_config(benchmark_dir + "/benchmark_targets.yaml")


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
    assert min(result) < targets["binning_1d"] * 1.25  # allows 25% error margin
    # update targets if > 20% improvement occurs beyond old bestmark
    if np.mean(result) < 0.8 * targets["binning_1d"]:
        print(f"Updating targets for 'binning_1d' to {float(np.mean(result))}")
        targets["binning_1d"] = float(np.mean(result))
        save_config(targets, benchmark_dir + "/benchmark_targets.yaml")


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
    assert min(result) < targets["binning_4d"] * 1.25  # allows 25% error margin
    # update targets if > 20% improvement occurs beyond old bestmark
    if np.mean(result) < 0.8 * targets["binning_4d"]:
        print(f"Updating targets for 'binning_4d' to {float(np.mean(result))}")
        targets["binning_4d"] = float(np.mean(result))
        save_config(targets, benchmark_dir + "/benchmark_targets.yaml")


def test_splinewarp() -> None:
    """Run a benchmark for the generation of the inverse dfield correction"""
    processor = SedProcessor(
        dataframe=dataframe.copy(),
        config=package_dir + "/config/mpes_example_config.yaml",
        folder_config={},
        user_config={},
        system_config={},
        verbose=True,
        verify_config=False,
    )
    processor.apply_momentum_correction()
    timer = timeit.Timer(
        "processor.mc.dfield_updated=True; processor.apply_momentum_correction()",
        globals={**globals(), **locals()},
    )
    result = timer.repeat(5, number=1)
    print(result)
    assert min(result) < targets["inv_dfield"] * 1.25  # allows 25% error margin
    # update targets if > 20% improvement occurs beyond old bestmark
    if np.mean(result) < 0.8 * targets["inv_dfield"]:
        print(f"Updating targets for 'inv_dfield' to {float(np.mean(result))}")
        targets["inv_dfield"] = float(np.mean(result))
        save_config(targets, benchmark_dir + "/benchmark_targets.yaml")


def test_workflow_1d() -> None:
    """Run a benchmark for 1d binning of converted data"""
    processor = SedProcessor(
        dataframe=dataframe.copy(),
        config=package_dir + "/config/mpes_example_config.yaml",
        folder_config={},
        user_config={},
        system_config={},
        verbose=True,
        verify_config=False,
    )
    processor.dataframe["sampleBias"] = 16.7
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
    assert min(result) < targets["workflow_1d"] * 1.25  # allows 25% error margin
    # update targets if > 20% improvement occurs beyond old bestmark
    if np.mean(result) < 0.8 * targets["workflow_1d"]:
        print(f"Updating targets for 'workflow_1d' to {float(np.mean(result))}")
        targets["workflow_1d"] = float(np.mean(result))
        save_config(targets, benchmark_dir + "/benchmark_targets.yaml")


def test_workflow_4d() -> None:
    """Run a benchmark for 4d binning of converted data"""
    processor = SedProcessor(
        dataframe=dataframe.copy(),
        config=package_dir + "/config/mpes_example_config.yaml",
        folder_config={},
        user_config={},
        system_config={},
        verbose=True,
        verify_config=False,
    )
    processor.dataframe["sampleBias"] = 16.7
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
    assert min(result) < targets["workflow_4d"] * 1.25  # allows 25% error margin
    # update targets if > 20% improvement occurs beyond old bestmark
    if np.mean(result) < 0.8 * targets["workflow_4d"]:
        print(f"Updating targets for 'workflow_4d' to {float(np.mean(result))}")
        targets["workflow_4d"] = float(np.mean(result))
        save_config(targets, benchmark_dir + "/benchmark_targets.yaml")


@pytest.mark.parametrize("loader", get_all_loaders())
def test_loader_compute(loader: BaseLoader) -> None:
    loader_name = get_loader_name_from_loader_object(loader)
    if loader.__name__ != "BaseLoader":
        if runs[loader_name] is None:
            pytest.skip("Not implemented")
        loaded_dataframe, _, loaded_metadata = loader.read_dataframe(
            runs=runs[loader_name],
            collect_metadata=False,
        )

        loaded_dataframe.compute()
        timer = timeit.Timer(
            "loaded_dataframe.compute()",
            globals={**globals(), **locals()},
        )
        result = timer.repeat(20, number=1)
        print(result)
        assert min(result) < targets[f"loader_compute_{loader_name}"] * 1.25  # allows 25% margin
        # update targets if > 20% improvement occurs beyond old bestmark
        if np.mean(result) < 0.8 * targets[f"loader_compute_{loader_name}"]:
            print(
                f"Updating targets for loader_compute_{loader_name}' "
                f"to {float(np.mean(result))}",
            )
            targets[f"loader_compute_{loader_name}"] = float(np.mean(result))
            save_config(targets, benchmark_dir + "/benchmark_targets.yaml")

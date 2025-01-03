"""Module tests.diagnostics, tests for the sed.diagnostics module
"""
from __future__ import annotations

import glob
import itertools
import os

import pytest

from sed.core.config import parse_config
from sed.diagnostics import grid_histogram
from sed.loader.loader_interface import get_loader

test_dir = os.path.dirname(__file__)
df_folder = f"{test_dir}/data/loader/mpes/"
calibration_folder = f"{test_dir}/data/calibrator/"
files = glob.glob(df_folder + "*.h5")
config = parse_config(
    f"{test_dir}/data/loader/mpes/config.yaml",
    folder_config={},
    user_config={},
    system_config={},
)
loader = get_loader("mpes", config=config)


@pytest.mark.parametrize(
    "ncols, backend",
    itertools.product([1, 2, 3, 4], ["matplotlib", "bokeh"]),
)
def test_plot_histogram(ncols: int, backend: str) -> None:
    """Test generation of data histogram

    Args:
        ncols (int): number of columns
        backend (str): plotting backend to use
    """
    dataframe, _, _ = loader.read_dataframe(files=files)
    axes = config["histogram"]["axes"]
    ranges = config["histogram"]["ranges"]
    bins = config["histogram"]["bins"]
    for loc, axis in enumerate(axes):
        if axis.startswith("@"):
            axes[loc] = config["dataframe"]["columns"].get(axis.strip("@"))
    values = {axis: dataframe[axis].compute() for axis in axes}
    grid_histogram(values, ncols, axes, bins, ranges, backend)

    # illegal keywords:
    with pytest.raises(TypeError):
        grid_histogram(values, ncols, axes, bins, ranges, backend, illegal_kwd=True)

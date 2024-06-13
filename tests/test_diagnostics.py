"""Module tests.diagnostics, tests for the sed.diagnostics module
"""
import glob
import itertools
import os
from importlib.util import find_spec

import pytest

from sed.core.config import parse_config
from sed.diagnostics import grid_histogram
from sed.loader.loader_interface import get_loader

#  pylint: disable=duplicate-code
package_dir = os.path.dirname(find_spec("sed").origin)
df_folder = package_dir + "/../tests/data/loader/mpes/"
folder = package_dir + "/../tests/data/calibrator/"
files = glob.glob(df_folder + "*.h5")
config = parse_config(package_dir + "/../tests/data/config/config.yaml")
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
            axes[loc] = config["dataframe"].get(axis.strip("@"))
    values = {axis: dataframe[axis].compute() for axis in axes}
    grid_histogram(values, ncols, axes, bins, ranges, backend)

import random

import numpy as np
import pytest

from sed.diagnostics import simulate_binned_data
from sed.io import load_tiff
from sed.io import to_tiff

shapes = []
for n in range(4):
    shapes.append([np.random.randint(10) + 1 for i in range(n)])
axes_names = ["x", "y", "t", "e"]
random.shuffle(axes_names)
binned_arrays = [simulate_binned_data(s, axes_names[: len(s)]) for s in shapes]


@pytest.mark.parametrize(
    "_da",
    binned_arrays,
    ids=lambda x: f"data_shape:{x.shape}",
)
def test_save_and_load_tiff_array(_da):
    axes_order = to_tiff(_da, "test.tiff", ret_axes_order=True)
    da_transposed = _da.transpose(*axes_order)
    as_array = load_tiff("test.tiff")
    assert np.allclose(da_transposed.values, as_array)


@pytest.mark.parametrize(
    "_da",
    binned_arrays,
    # ids=lambda x: f"input_data_shape:{x.shape}",
)
def test_save_and_load_tiff_xarray(_da):
    axes_order = to_tiff(_da, "test.tiff", ret_axes_order=True)
    da_transposed = _da.transpose(*axes_order)
    as_xarray = load_tiff("test.tiff")
    assert np.allclose(da_transposed, as_xarray)
    for k, v in _da.coords.items():
        assert np.allclose(as_xarray.coords[k], v)

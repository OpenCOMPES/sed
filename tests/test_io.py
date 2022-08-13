"""This is a code that performs several tests for the input/output functions
"""
import random

import numpy as np
import pytest
import xarray as xr

from sed.diagnostics import simulate_binned_data
from sed.io import load_h5
from sed.io import load_tiff
from sed.io import to_h5
from sed.io import to_tiff

shapes = []
for n in range(4):
    shapes.append(tuple(np.random.randint(10) + 2 for i in range(n + 1)))
axes_names = ["x", "y", "delay", "e"]
random.shuffle(axes_names)
binned_arrays = [simulate_binned_data(s, axes_names[: len(s)]) for s in shapes]


@pytest.mark.parametrize(
    "_da",
    binned_arrays,
    ids=lambda x: f"ndims:{len(x.shape)}",
)
def test_save_and_load_tiff_array(_da):
    """Test the tiff saving/loading function for np.ndarrays."""
    nd_array = _da.data
    if nd_array.ndim > 1:
        to_tiff(nd_array, "test")
        as_array = load_tiff("test.tiff")
        np.testing.assert_allclose(nd_array, as_array)


@pytest.mark.parametrize(
    "_da",
    binned_arrays,
    ids=lambda x: f"ndims:{len(x.shape)}",
)
def test_save_and_load_tiff_xarray(_da):
    """Test the tiff saving/loading function for xr.DataArrays."""
    axes_order = to_tiff(_da, "test")
    da_transposed = _da.transpose(*axes_order)
    as_xarray = load_tiff("test.tiff", coords=_da.coords, dims=axes_order)
    np.testing.assert_allclose(da_transposed, as_xarray)
    for k, v in _da.coords.items():
        np.testing.assert_allclose(as_xarray.coords[k], v)


@pytest.mark.parametrize(
    "_da",
    binned_arrays,
    ids=lambda x: f"ndims:{len(x.shape)}",
)
def test_save_and_load_hdf5(_da):
    """Test the hdf5 saving/loading function."""
    faddr = "test.h5"
    to_h5(_da, faddr, mode="w")
    loaded = load_h5(faddr)
    xr.testing.assert_equal(_da, loaded)

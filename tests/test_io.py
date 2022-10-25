"""This is a code that performs several tests for the input/output functions
"""
import random
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from sed.io.hdf5 import load_h5
from sed.io.hdf5 import to_h5
from sed.io.tiff import _sort_dims_for_imagej
from sed.io.tiff import load_tiff
from sed.io.tiff import to_tiff
from tests.helpers import simulate_binned_data

shapes = []
for n in range(4):
    shapes.append(tuple(np.random.randint(10) + 2 for i in range(n + 1)))
axes_names = ["X", "Y", "T", "E"]
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
def test_save_xarr_to_tiff(_da):
    """Test the tiff saving function for xr.DataArrays."""
    to_tiff(_da, "test")
    assert Path("test.tiff").is_file()


@pytest.mark.parametrize(
    "_da",
    binned_arrays,
    ids=lambda x: f"ndims:{len(x.shape)}",
)
def test_save_and_load_tiff_xarray(_da):
    """Test the tiff saving/loading function for xr.DataArrays."""
    to_tiff(_da, "test")
    loaded = load_tiff("test.tiff")
    dims_order = _sort_dims_for_imagej(_da.dims)
    transposed = _da.transpose(*dims_order).astype(np.float32)
    np.testing.assert_allclose(
        transposed.values,
        loaded.values,
    )


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
    np.testing.assert_equal(_da.attrs, loaded.attrs)
    for axis in _da.coords:
        np.testing.assert_equal(_da[axis].attrs, loaded[axis].attrs)

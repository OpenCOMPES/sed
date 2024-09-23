"""This file contains code that performs several tests for the input/output functions
"""
from __future__ import annotations

import os
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
from .helpers import simulate_binned_data  # noreorder

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
def test_save_and_load_tiff_array(_da: xr.DataArray) -> None:
    """Test the tiff saving/loading function for np.ndarrays.

    Args:
        _da (xr.DataArray): binned DataArray
    """
    nd_array = _da.data
    if nd_array.ndim > 1:
        to_tiff(nd_array, f"test1{nd_array.ndim}")
        as_array = load_tiff(f"test1{nd_array.ndim}.tiff")
        np.testing.assert_allclose(nd_array, as_array)
        os.remove(f"test1{nd_array.ndim}.tiff")


@pytest.mark.parametrize(
    "_da",
    binned_arrays,
    ids=lambda x: f"ndims:{len(x.shape)}",
)
def test_save_xarr_to_tiff(_da: xr.DataArray) -> None:
    """Test the tiff saving function for xr.DataArrays.

    Args:
        _da (xr.DataArray): binned DataArray
    """
    to_tiff(_da, f"test2{len(_da.shape)}")
    assert Path(f"test2{len(_da.shape)}.tiff").is_file()
    os.remove(f"test2{len(_da.shape)}.tiff")


@pytest.mark.parametrize(
    "_da",
    binned_arrays,
    ids=lambda x: f"ndims:{len(x.shape)}",
)
def test_save_and_load_tiff_xarray(_da: xr.DataArray) -> None:
    """Test the tiff saving/loading function for xr.DataArrays.

    rgs:
        _da (xr.DataArray): binned DataArray
    """
    to_tiff(_da, f"test3{len(_da.shape)}")
    loaded = load_tiff(f"test3{len(_da.shape)}.tiff")
    dims_order = _sort_dims_for_imagej(_da.dims)
    transposed = _da.transpose(*dims_order).astype(np.float32)
    np.testing.assert_allclose(
        transposed.values,
        loaded.values,
    )
    os.remove(f"test3{len(_da.shape)}.tiff")


@pytest.mark.parametrize(
    "_da",
    binned_arrays,
    ids=lambda x: f"ndims:{len(x.shape)}",
)
def test_save_and_load_hdf5(_da: xr.DataArray) -> None:
    """Test the hdf5 saving/loading function.

    Args:
        _da (xr.DataArray): binned DataArray
    """
    faddr = f"test{len(_da.shape)}.h5"
    to_h5(_da, faddr, mode="w")
    loaded = load_h5(faddr)
    xr.testing.assert_equal(_da, loaded)
    np.testing.assert_equal(_da.attrs, loaded.attrs)
    for axis in _da.coords:
        np.testing.assert_equal(_da[axis].attrs, loaded[axis].attrs)
    os.remove(faddr)

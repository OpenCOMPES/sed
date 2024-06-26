"""This file contains code that performs several tests for the sed.binning module
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from typing import cast

import dask.dataframe as ddf
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sed.binning.binning import bin_dataframe
from sed.binning.binning import bin_partition
from sed.binning.binning import normalization_histogram_from_timed_dataframe
from sed.binning.binning import normalization_histogram_from_timestamps
from sed.binning.binning import numba_histogramdd
from sed.binning.binning import simplify_binning_arguments
from sed.binning.numba_bin import _hist_from_bin_range
from sed.binning.numba_bin import _hist_from_bins
from sed.binning.numba_bin import binsearch
from sed.binning.utils import bin_centers_to_bin_edges
from sed.binning.utils import bin_edges_to_bin_centers
from .helpers import get_linear_bin_edges  # noreorder

sample = np.random.randn(int(1e5), 3)
columns = ["x", "y", "z"]
sample_pdf = pd.DataFrame(sample, columns=columns)
sample_ddf = ddf.from_pandas(sample_pdf, npartitions=10)
bins: Sequence[int] = tuple(np.random.randint(5, 50, size=3, dtype=int))
ranges_array = 0.5 + np.random.rand(6).reshape(3, 2)
ranges_array[:, 0] = -ranges_array[:, 0]
ranges: Sequence[tuple] = tuple(tuple(r) for r in ranges_array)

arrays = [get_linear_bin_edges(b, r) for r, b in zip(ranges, bins)]

sample_int = np.random.randint(
    low=60000,
    high=70001,
    size=(int(1e5), 3),
    dtype=int,
)
bins_int = tuple(np.random.randint(10, 300, size=3, dtype=int))
ranges_int = [tuple(np.sort(np.random.randint(60000, 70000, size=2, dtype=int))) for _ in range(3)]
arrays_int = [get_linear_bin_edges(b, r) for r, b in zip(ranges_int, bins_int)]

bins_round = [300]
HALFBINSIZE = (65000.0 - 66600.0) / 300 / 2
ranges_round = [(65000.0 - HALFBINSIZE, 66600.0 - HALFBINSIZE)]
arrays_round = [get_linear_bin_edges(b, r) for r, b in zip(ranges_round, bins_round)]


@pytest.mark.parametrize(
    "_samples",
    [sample[:, :1], sample[:, :2], sample[:, :3]],
    ids=lambda x: f"samples:{x.shape}",
)
@pytest.mark.parametrize(
    "_bins",
    # [tuple(bins[:i+1]) for i in range(3)],
    [bins[:1], bins[:2], bins[:3]],
    ids=lambda x: f"bins:{len(x)}",
)
def test_histdd_error_is_raised(_samples: np.ndarray, _bins: list[int]) -> None:
    """Test if the correct error is raised if the bins and sample shapes do not match

    Args:
        _samples (np.ndarray): Samples array
        _bins (list[int]): Bins list
    """
    with pytest.raises(ValueError):
        if _samples.shape[1] == len(_bins):
            pytest.skip("Not of interest")

        _hist_from_bin_range.py_func(_samples, _bins, np.array([ranges[0]]))


@pytest.mark.parametrize(
    "args",
    [
        (sample[:, :1], arrays[:1], 1),
        (sample[:, :2], arrays[:2], 2),
        (sample[:, :3], arrays[:3], 3),
        (sample_int[:, :1], arrays_int[:1], 4),
        (sample_int[:, :2], arrays_int[:2], 5),
        (sample_int[:, :3], arrays_int[:3], 6),
        (sample_int[:, :1], arrays_round[:1], 7),
    ],
    ids=lambda x: f"ndim: {x[2]}"
    if x[2] < 4
    else f"ndim: {x[2]-3}-int"
    if x[2] < 7
    else f"ndim: {x[2]-6}-round",
)
def test_histdd_bins_as_numpy(args: tuple[np.ndarray, np.ndarray, int]) -> None:
    """Test whether the numba_histogramdd functions produces the same result
    as np.histogramdd if called with a list of bin edges

    Args:
        args (tuple[np.ndarray, np.ndarray, int]): Tuple of
            (samples, bin_edges, dimension)
    """
    sample_, bins_, _ = args
    hist1, edges1 = np.histogramdd(sample_, bins_)
    hist2, edges2 = numba_histogramdd(sample_, bins_)
    np.testing.assert_allclose(hist1, hist2)
    for edges1_, edges2_ in zip(edges1, edges2):
        np.testing.assert_allclose(edges1_, edges2_)


@pytest.mark.parametrize(
    "args",
    [
        (sample[:, :1], bins[:1], ranges[:1], 1),
        (sample[:, :2], bins[:2], ranges[:2], 2),
        (sample[:, :3], bins[:3], ranges[:3], 3),
        (sample_int[:, :1], bins_int[:1], ranges_int[:1], 4),
        (sample_int[:, :2], bins_int[:2], ranges_int[:2], 5),
        (sample_int[:, :3], bins_int[:3], ranges_int[:3], 6),
        (sample_int[:, :1], bins_round[:1], ranges_round[:1], 7),
    ],
    ids=lambda x: f"ndim: {x[3]}"
    if x[3] < 4
    else f"ndim: {x[3]-3}-int"
    if x[3] < 7
    else f"ndim: {x[3]-6}-round",
)
def test_histdd_ranges_as_numpy(args: tuple[np.ndarray, tuple, tuple, int]) -> None:
    """Test whether the numba_histogramdd functions produces the same result
    as np.histogramdd if called with bin numbers and ranges

    Args:
        args (tuple[np.ndarray, np.ndarray, np.ndarray, int]): Tuple of
            (samples, bins, ranges, dimension)
    """
    sample_, bins_, ranges_, _ = args
    hist1, edges1 = np.histogramdd(sample_, bins_, ranges_)
    hist2, edges2 = numba_histogramdd(sample_, bins_, ranges_)
    np.testing.assert_allclose(hist1, hist2)
    for edges1_, edges2_ in zip(edges1, edges2):
        np.testing.assert_allclose(edges1_, edges2_)


@pytest.mark.parametrize(
    "args",
    [
        (sample[:, :1], bins[0], ranges[:1], 1),
        (sample[:, :2], bins[0], ranges[:2], 2),
        (sample[:, :3], bins[0], ranges[:3], 3),
        (sample_int[:, :1], bins_int[0], ranges_int[:1], 4),
        (sample_int[:, :2], bins_int[0], ranges_int[:2], 5),
        (sample_int[:, :3], bins_int[0], ranges_int[:3], 6),
        (sample_int[:, :1], bins_round[0], ranges_round[:1], 7),
    ],
    ids=lambda x: f"ndim: {x[3]}"
    if x[3] < 4
    else f"ndim: {x[3]-3}-int"
    if x[3] < 7
    else f"ndim: {x[3]-6}-round",
)
def test_histdd_one_bins_as_numpy(args: tuple[np.ndarray, int, tuple, int]) -> None:
    """Test whether the numba_histogramdd functions produces the same result
    as np.histogramdd if called with bin numbers and ranges

    Args:
        args (tuple[np.ndarray, np.ndarray, np.ndarray, int]): Tuple of
            (samples, bins, ranges, dimension)
    """
    sample_, bins_, ranges_, _ = args
    hist1, edges1 = np.histogramdd(sample_, bins_, ranges_)
    hist2, edges2 = numba_histogramdd(sample_, bins_, ranges_)
    np.testing.assert_allclose(hist1, hist2)
    for edges1_, edges2_ in zip(edges1, edges2):
        np.testing.assert_allclose(edges1_, edges2_)


@pytest.mark.parametrize(
    "args",
    [
        (sample[:, :1], bins[:1], ranges[:1], arrays[:1], 1),
        (sample[:, :2], bins[:2], ranges[:2], arrays[:2], 2),
        (sample[:, :3], bins[:3], ranges[:3], arrays[:3], 3),
        (sample_int[:, :1], bins_int[:1], ranges_int[:1], arrays_int[:1], 4),
        (sample_int[:, :2], bins_int[:2], ranges_int[:2], arrays_int[:2], 5),
        (sample_int[:, :3], bins_int[:3], ranges_int[:3], arrays_int[:3], 6),
        (sample_int[:, :1], bins_round[:1], ranges_round[:1], arrays_round[:1], 7),
    ],
    ids=lambda x: f"ndim: {x[4]}"
    if x[4] < 4
    else f"ndim: {x[4]-3}-int"
    if x[4] < 7
    else f"ndim: {x[4]-6}-round",
)
def test_from_bins_equals_from_bin_range(
    args: tuple[np.ndarray, int, tuple, np.ndarray, int],
) -> None:
    """Test whether the numba_histogramdd functions produces the same result
    if called with bin numbers and ranges or with bin edges.

    Args:
        args (tuple[np.ndarray, int, tuple, np.ndarray, int]): Tuple of
            (samples, bins, ranges, bin_edges, dimension)
    """
    sample_, bins_, ranges_, arrays_, _ = args
    hist1, edges1 = numba_histogramdd(sample_, bins_, ranges_)
    hist2, edges2 = numba_histogramdd(sample_, arrays_)
    np.testing.assert_allclose(hist1, hist2, verbose=True)
    for edges1_, edges2_ in zip(edges1, edges2):
        np.testing.assert_allclose(edges1_, edges2_)


@pytest.mark.parametrize(
    "args",
    [
        (sample[:, :1], arrays[:1], 1),
    ],
    ids=lambda x: f"ndim: {x[2]}",
)
def test_numba_hist_from_bins(args: tuple[np.ndarray, np.ndarray, int]) -> None:
    """Run tests using the _hist_from_bins function without numba jit.

    Args:
        args (tuple[np.ndarray, np.ndarray, int]): Tuple of
            (samples, bin_edges, dimension)
    """
    sample_, arrays_, _ = args
    with pytest.raises(ValueError):
        _hist_from_bins.py_func(
            sample_,
            arrays_[0],
            tuple(b.size - 1 for b in arrays_),
        )
    _, edges = numba_histogramdd(sample_, arrays_)
    _hist_from_bins.py_func(
        sample_,
        arrays_,
        tuple(b.size - 1 for b in arrays_),
    )
    assert binsearch.py_func(edges[0], 0) > -1
    assert binsearch.py_func(edges[0], -100) == -1
    assert binsearch.py_func(edges[0], np.nan) == -1
    assert binsearch.py_func(edges[0], 100) == -1
    assert binsearch.py_func(edges[0], edges[0][-1]) == len(edges[0]) - 2


@pytest.mark.parametrize(
    "args",
    [
        (sample[:, :1], bins[:1], ranges[:1], 1),
    ],
    ids=lambda x: f"ndim: {x[3]}",
)
def test_numba_hist_from_bins_ranges(args: tuple[np.ndarray, int, tuple, int]) -> None:
    """Run tests using the _hist_from_bins_ranges function without numba jit.

    Args:
        args (tuple[np.ndarray, int, tuple, int]): Tuple of
            (samples, bins, ranges, dimension)
    """
    sample_, bins_, ranges_, _ = args
    _hist_from_bin_range.py_func(sample_, bins_, np.asarray(ranges_))


def test_bin_centers_to_bin_edges() -> None:
    """Test the conversion from bin centers to bin edges"""
    stepped_array = np.concatenate(
        [
            arrays[0],
            arrays[1][1:] + arrays[0][-1] - arrays[1][0],
            arrays[2][1:] + arrays[0][-1] + arrays[1][-1] - arrays[2][0] - arrays[1][0],
        ],
    )
    bin_edges = bin_centers_to_bin_edges(stepped_array)
    for i in range(0, (len(bin_edges) - 1)):
        assert bin_edges[i] < stepped_array[i]
        assert bin_edges[i + 1] > stepped_array[i]


def test_bin_edges_to_bin_centers() -> None:
    """Test the conversion from bin edges to bin centers"""
    stepped_array = np.concatenate(
        [
            arrays[0],
            arrays[1][1:] + arrays[0][-1] - arrays[1][0],
            arrays[2][1:] + arrays[0][-1] + arrays[1][-1] - arrays[2][0] - arrays[1][0],
        ],
    )
    bin_centers = bin_edges_to_bin_centers(stepped_array)
    for i in range(0, (len(bin_centers) - 1)):
        assert stepped_array[i] < bin_centers[i]
        assert stepped_array[i + 1] > bin_centers[i]


@pytest.mark.parametrize(
    "args",
    [
        (bins[:1], columns[:1], ranges[:1], 1),
        (bins[:2], columns[:2], ranges[:2], 2),
        (bins[:3], columns[:3], ranges[:3], 3),
    ],
    ids=lambda x: f"ndim: {x[3]}",
)
@pytest.mark.parametrize(
    "arg_type",
    [
        "int",
        "list_int",
        "array",
        "tuple",
        "dict_int",
        "dict_tuple",
        "dict_array",
    ],
)
def test_simplify_binning_arguments(
    args: tuple[list[int], list[str], list[tuple[float, float]]],
    arg_type: str,
) -> None:
    """Test the result of the _simplify_binning_arguments functions for number of
    bins and ranges
    """
    bins_: int | list | dict = None
    axes_: list[str] = None
    ranges_: list[tuple[float, float]] = None
    bins_expected: list[Any] = None
    axes_expected: list[Any] = None
    ranges_expected: list[Any] = None

    bin_centers = []
    for i in range(len(args[1])):
        bin_centers.append(
            np.linspace(args[2][i][0], args[2][i][1], args[0][i] + 1),
        )

    if arg_type == "int":
        bins_ = args[0][0]
        axes_ = args[1]
        ranges_ = args[2]
        bins_expected = [bins_] * len(args[0])
        axes_expected = axes_
        ranges_expected = ranges_
    elif arg_type == "list_int":
        bins_ = args[0]
        axes_ = args[1]
        ranges_ = args[2]
        bins_expected = bins_
        axes_expected = axes_
        ranges_expected = ranges_
    elif arg_type == "array":
        bins_ = []
        for i in range(len(args[0])):
            bins_.append(bin_centers[i])
        axes_ = args[1]
        bins_expected = bins_
        axes_expected = axes_
    elif arg_type == "tuple":
        bins_ = []
        for i in range(len(args[0])):
            bins_.append((args[2][i][0], args[2][i][1], args[0][i]))
        axes_ = args[1]
        bins_expected = args[0]
        axes_expected = axes_
        ranges_expected = args[2]
    elif arg_type == "dict_int":
        bins_ = {}
        for i, axis in enumerate(args[1]):
            bins_[axis] = args[0][i]
        ranges_ = args[2]
        bins_expected = args[0]
        axes_expected = args[1]
        ranges_expected = args[2]
    elif arg_type == "dict_array":
        bins_ = {}
        for i, axis in enumerate(args[1]):
            bins_[axis] = bin_centers[i]
        bins_expected = bin_centers
        axes_expected = args[1]
    elif arg_type == "dict_tuple":
        bins_ = {}
        for i, axis in enumerate(args[1]):
            bins_[axis] = (args[2][i][0], args[2][i][1], args[0][i])
        bins_expected = args[0]
        axes_expected = args[1]
        ranges_expected = args[2]

    bins__, axes__, ranges__ = simplify_binning_arguments(
        bins_,
        axes_,
        ranges_,
    )

    for i, _ in enumerate(bins__):
        np.testing.assert_array_equal(bins__[i], bins_expected[i])
        np.testing.assert_array_equal(axes__[i], axes_expected[i])
        if ranges__ is not None:
            np.testing.assert_array_equal(ranges__[i], ranges_expected[i])


def test_bin_partition() -> None:
    """Test bin_partition function"""
    # test skipping checks in bin_partition
    with pytest.raises(TypeError):
        res = bin_partition(
            part=sample_pdf,
            bins=[10, np.array([10, 20])],  # type: ignore[arg-type]
            axes=columns,
            ranges=ranges,
            skip_test=True,
        )
    with pytest.raises(TypeError):
        res = bin_partition(
            part=sample_pdf,
            bins=bins,
            axes=["x", 10],  # type: ignore[arg-type, list-item]
            ranges=ranges,
            skip_test=True,
        )
    # binning with bin numbers
    res1, edges1 = bin_partition(
        part=sample_pdf,
        bins=bins,
        axes=columns,
        ranges=ranges,
        skip_test=False,
        return_edges=True,
    )
    assert isinstance(res1, np.ndarray)
    # binning with bin centers
    bin_centers = [np.linspace(r[0], r[1], b, endpoint=False) for r, b in zip(ranges, bins)]

    res2, edges2 = bin_partition(
        part=sample_pdf,
        bins=bin_centers,
        axes=columns,
        skip_test=False,
        return_edges=True,
    )
    assert isinstance(res2, np.ndarray)
    np.testing.assert_allclose(res1, res2)
    for edge1, edge2 in zip(edges1, edges2):
        np.testing.assert_allclose(edge1, edge2)

    # test jittering, list
    res = bin_partition(
        part=sample_pdf,
        bins=bins,
        axes=columns,
        ranges=ranges,
        skip_test=False,
        jitter=columns,
    )
    assert not np.allclose(cast(np.ndarray, res), res1)
    # test jittering, dict
    res = bin_partition(
        part=sample_pdf,
        bins=bins,
        axes=columns,
        ranges=ranges,
        skip_test=False,
        jitter={axis: {"amplitude": 0.5, "mode": "normal"} for axis in columns},
    )
    assert not np.allclose(cast(np.ndarray, res), res1)
    # numpy mode
    with pytest.raises(ValueError):
        res = bin_partition(
            part=sample_pdf,
            bins=bins,
            axes=columns,
            ranges=ranges,
            skip_test=False,
            hist_mode="invalid",
        )
    res = bin_partition(
        part=sample_pdf,
        bins=bins,
        axes=columns,
        ranges=ranges,
        skip_test=False,
        hist_mode="numpy",
    )
    assert np.allclose(cast(np.ndarray, res), res1)


def test_non_numeric_dtype_error() -> None:
    """Test bin_partition function"""
    pdf = sample_pdf.astype({"x": "string", "y": "int32", "z": "int32"})
    with pytest.raises(ValueError) as err:
        _ = bin_partition(
            part=pdf,
            bins=bins,  # type: ignore[arg-type]
            axes=columns,
            ranges=ranges,
            skip_test=False,
        )
    assert "Encountered data types were ['x: string', 'y: int32', 'z: int32']" in str(err.value)


def test_bin_dataframe() -> None:
    """Test bin_dataframe function"""
    res = bin_dataframe(df=sample_ddf, bins=bins, axes=columns, ranges=ranges)
    assert isinstance(res, xr.DataArray)
    for i, axis in enumerate(columns):
        assert len(res.coords[axis]) == bins[i]
    # binning with bin centers
    bin_centers = [np.linspace(r[0], r[1], b, endpoint=False) for r, b in zip(ranges, bins)]
    res = bin_dataframe(df=sample_ddf, bins=bin_centers, axes=columns)
    assert isinstance(res, xr.DataArray)
    for i, axis in enumerate(columns):
        assert len(res.coords[axis]) == len(bin_centers[i])
    # legacy modes
    with pytest.raises(ValueError):
        res = bin_dataframe(df=sample_ddf, bins=bins, axes=columns, ranges=ranges, mode="invalid")
    res2 = bin_dataframe(df=sample_ddf, bins=bins, axes=columns, ranges=ranges, mode="legacy")
    np.testing.assert_allclose(res.values, res2.values)
    res2 = bin_dataframe(df=sample_ddf, bins=bins, axes=columns, ranges=ranges, mode="lean")
    np.testing.assert_allclose(res.values, res2.values)


def test_normalization_histogram_from_timestamps() -> None:
    """Test the function to generate the normalization histogram from timestamps"""
    time_stamped_df = sample_ddf.copy()
    time_stamped_df["timeStamps"] = time_stamped_df.index
    res = bin_dataframe(df=sample_ddf, bins=[bins[0]], axes=[columns[0]], ranges=[ranges[0]])
    histogram = normalization_histogram_from_timestamps(
        df=time_stamped_df,
        axis=columns[0],
        bin_centers=res.coords[columns[0]].values,
        time_stamp_column="timeStamps",
    )
    np.testing.assert_allclose(res / res.sum(), histogram / histogram.sum(), rtol=0.01)


def test_normalization_histogram_from_timed_dataframe() -> None:
    """Test the function to generate the normalization histogram from the timed dataframe"""
    res = bin_dataframe(df=sample_ddf, bins=[bins[0]], axes=[columns[0]], ranges=[ranges[0]])
    histogram = normalization_histogram_from_timed_dataframe(
        df=sample_ddf,
        axis=columns[0],
        bin_centers=res.coords[columns[0]].values,
        time_unit=1,
    )
    np.testing.assert_allclose(res / res.sum(), histogram / histogram.sum())

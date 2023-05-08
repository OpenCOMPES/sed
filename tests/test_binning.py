"""This file contains code that performs several tests for the sed.binning module
"""
from typing import Any
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import pytest

from sed.binning.binning import numba_histogramdd
from sed.binning.binning import simplify_binning_arguments
from sed.binning.numba_bin import _hist_from_bin_range
from sed.binning.utils import bin_centers_to_bin_edges
from sed.binning.utils import bin_edges_to_bin_centers
from .helpers import get_linear_bin_edges

sample = np.random.randn(int(1e5), 3)
columns = ["x", "y", "z"]
sample_df = pd.DataFrame(sample, columns=columns)
bins: Sequence[int] = tuple(np.random.randint(5, 50, size=3, dtype=int))
ranges_array = 0.5 + np.random.rand(6).reshape(3, 2)
ranges_array[:, 0] = -ranges_array[:, 0]
ranges: Sequence[tuple] = tuple(tuple(r) for r in ranges_array)

arrays = [get_linear_bin_edges(b, r) for r, b in zip(ranges, bins)]

sample_int = np.random.randint(
    low=64000,
    high=67000,
    size=(int(1e5), 3),
    dtype=int,
)
bins_int = (300, 50, 100)
ranges_int = [(65000.0, 66600.0), (65000.0, 70000.0), (66000.0, 660020.0)]

arrays_int = [get_linear_bin_edges(b, r) for r, b in zip(ranges_int, bins_int)]


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
def test_histdd_error_is_raised(_samples: np.ndarray, _bins: List[int]):
    """Test if the correct error is raised if the bins and sample shapes do not match

    Args:
        _samples (np.ndarray): Samples array
        _bins (List[Tuple]): Bins list
    """
    with pytest.raises(ValueError):
        if _samples.shape[1] == len(_bins):
            pytest.skip("Not of interest")

        _hist_from_bin_range(_samples, _bins, np.array([ranges[0]]))


@pytest.mark.parametrize(
    "args",
    [
        (sample[:, :1], arrays[:1], 1),
        (sample[:, :2], arrays[:2], 2),
        (sample[:, :3], arrays[:3], 3),
        (sample_int[:, :1], arrays_int[:1], 4),
        (sample_int[:, :2], arrays_int[:2], 5),
        (sample_int[:, :3], arrays_int[:3], 6),
    ],
    ids=lambda x: f"ndim: {x[2]}" if x[2] < 4 else f"ndim: {x[2]-3}-int",
)
def test_histdd_bins_as_numpy(args: Tuple[np.ndarray, np.ndarray, int]):
    """Test whether the numba_histogramdd functions produces the same result
    as np.histogramdd if called with a list of bin edgees

    Args:
        args (Tuple[np.ndarray, np.ndarray, int]): Tuple of
            (samples, bin_edges, dimension)
    """
    sample_, bins_, _ = args
    hist1, edges1 = np.histogramdd(sample_, bins_)
    hist2, edges2 = numba_histogramdd(sample_, bins_)
    np.testing.assert_allclose(hist1, hist2)
    for (edges1_, edges2_) in zip(edges1, edges2):
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
    ],
    ids=lambda x: f"ndim: {x[3]}" if x[3] < 4 else f"ndim: {x[3]-3}-int",
)
def test_histdd_ranges_as_numpy(args: Tuple[np.ndarray, tuple, tuple, int]):
    """Test whether the numba_histogramdd functions produces the same result
    as np.histogramdd if called with bin numbers and ranges

    Args:
        args (Tuple[np.ndarray, np.ndarray, np.ndarray, int]): Tuple of
            (samples, bins, ranges, dimension)
    """
    sample_, bins_, ranges_, _ = args
    hist1, edges1 = np.histogramdd(sample_, bins_, ranges_)
    hist2, edges2 = numba_histogramdd(sample_, bins_, ranges_)
    np.testing.assert_allclose(hist1, hist2)
    for (edges1_, edges2_) in zip(edges1, edges2):
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
    ],
    ids=lambda x: f"ndim: {x[3]}" if x[3] < 4 else f"ndim: {x[3]-3}-int",
)
def test_histdd_one_bins_as_numpy(args: Tuple[np.ndarray, int, tuple, int]):
    """Test whether the numba_histogramdd functions produces the same result
    as np.histogramdd if called with bin numbers and ranges

    Args:
        args (Tuple[np.ndarray, np.ndarray, np.ndarray, int]): Tuple of
            (samples, bins, ranges, dimension)
    """
    sample_, bins_, ranges_, _ = args
    hist1, edges1 = np.histogramdd(sample_, bins_, ranges_)
    hist2, edges2 = numba_histogramdd(sample_, bins_, ranges_)
    np.testing.assert_allclose(hist1, hist2)
    for (edges1_, edges2_) in zip(edges1, edges2):
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
    ],
    ids=lambda x: f"ndim: {x[4]}" if x[4] < 4 else f"ndim: {x[4]-3}-int",
)
def test_from_bins_equals_from_bin_range(
    args: Tuple[np.ndarray, int, tuple, np.ndarray, int],
):
    """Test whether the numba_histogramdd functions produces the same result
    if called with bin numbers and ranges or with bin edges.

    Args:
        args (Tuple[np.ndarray, int, tuple, np.ndarray, int]): Tuple of
            (samples, bins, ranges, bin_edges, dimension)
    """
    sample_, bins_, ranges_, arrays_, _ = args
    hist1, edges1 = numba_histogramdd(sample_, bins_, ranges_)
    hist2, edges2 = numba_histogramdd(sample_, arrays_)
    np.testing.assert_allclose(hist1, hist2)
    for (edges1_, edges2_) in zip(edges1, edges2):
        np.testing.assert_allclose(edges1_, edges2_)


def test_bin_centers_to_bin_edges():
    """Test the conversion from bin centers to bin edges"""
    stepped_array = np.concatenate(
        [
            arrays[0],
            arrays[1][1:] + arrays[0][-1] - arrays[1][0],
            arrays[2][1:]
            + arrays[0][-1]
            + arrays[1][-1]
            - arrays[2][0]
            - arrays[1][0],
        ],
    )
    bin_edges = bin_centers_to_bin_edges(stepped_array)
    for i in range(0, (len(bin_edges) - 1)):
        assert bin_edges[i] < stepped_array[i]
        assert bin_edges[i + 1] > stepped_array[i]


def test_bin_edges_to_bin_centers():
    """Test the conversion from bin edges to bin centers"""
    stepped_array = np.concatenate(
        [
            arrays[0],
            arrays[1][1:] + arrays[0][-1] - arrays[1][0],
            arrays[2][1:]
            + arrays[0][-1]
            + arrays[1][-1]
            - arrays[2][0]
            - arrays[1][0],
        ],
    )
    bin_centers = bin_edges_to_bin_centers(stepped_array)
    for i in range(0, (len(bin_centers) - 1)):
        assert stepped_array[i] < bin_centers[i]
        assert stepped_array[i + 1] > bin_centers[i]


bins = [10, 20, 30]
axes = ["a", "b", "c"]
ranges = [(-1, 1), (-2, 2), (-3, 3)]


@pytest.mark.parametrize(
    "args",
    [
        (bins[:1], axes[:1], ranges[:1], 1),
        (bins[:2], axes[:2], ranges[:2], 2),
        (bins[:3], axes[:3], ranges[:3], 3),
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
    args: Tuple[List[int], List[str], List[Tuple[float, float]]],
    arg_type: str,
):
    """Test the result of the _simplify_binning_arguments functions for number of
    bins and ranges
    """
    bins_: Union[int, list, dict] = None
    axes_: List[str] = None
    ranges_: List[Tuple[float, float]] = None
    bins_expected: List[Any] = None
    axes_expected: List[Any] = None
    ranges_expected: List[Any] = None

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

    for i, bin_ in enumerate(bins__):
        np.testing.assert_array_equal(bin_, bins_expected[i])
        np.testing.assert_array_equal(axes__[i], axes_expected[i])
        if ranges__ is not None:
            np.testing.assert_array_equal(ranges__[i], ranges_expected[i])

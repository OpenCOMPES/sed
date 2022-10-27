"""This file contains code that performs several tests for the sed.binning module
"""
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from sed.binning.binning import _simplify_binning_arguments
from sed.binning.binning import numba_histogramdd
from sed.binning.numba_bin import _hist_from_bin_range
from sed.binning.utils import bin_centers_to_bin_edges
from sed.binning.utils import bin_edges_to_bin_centers
from .helpers import get_linear_bin_edges

sample = np.random.randn(int(1e2), 3)
columns = ["x", "y", "z"]
sample_df = pd.DataFrame(sample, columns=columns)
bins = tuple(np.random.randint(5, 50, size=3))
ranges = 0.5 + np.random.rand(6).reshape(3, 2)
ranges[:, 0] = -ranges[:, 0]
ranges = tuple(tuple(r) for r in ranges)

arrays = [
    get_linear_bin_edges(np.linspace(r[0], r[1], b))
    for r, b in zip(ranges, bins)
]


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
    ],
    ids=lambda x: f"ndim: {x[2]}",
)
def test_histdd_bins_as_numpy(args: Tuple[np.ndarray, np.ndarray, int]):
    """Test whether the numba_histogramdd functions produces the same result
    as np.histogramdd if called with a list of bin edgees

    Args:
        args (Tuple[np.ndarray, np.ndarray, int]):
        Tuple of (samples, bin_edges, dimension)
    """
    sample_, bins_, _ = args
    hist1, _ = np.histogramdd(sample_, bins_)
    hist2, _ = numba_histogramdd(sample_, bins_)
    np.testing.assert_allclose(hist1, hist2)


@pytest.mark.parametrize(
    "args",
    [
        (sample[:, :1], bins[:1], ranges[:1], 1),
        (sample[:, :2], bins[:2], ranges[:2], 2),
        (sample[:, :3], bins[:3], ranges[:3], 3),
    ],
    ids=lambda x: f"ndim: {x[3]}",
)
def test_histdd_ranges_as_numpy(args: Tuple[np.ndarray, tuple, tuple, int]):
    """Test whether the numba_histogramdd functions produces the same result
    as np.histogramdd if called with bin numbers and ranges

    Args:
        args (Tuple[np.ndarray, np.ndarray, np.ndarray, int]):
        Tuple of (samples, bins, ranges, dimension)
    """
    sample_, bins_, ranges_, _ = args
    hist1, _ = np.histogramdd(sample_, bins_, ranges_)
    hist2, _ = numba_histogramdd(sample_, bins_, ranges_)
    np.testing.assert_allclose(hist1, hist2)


@pytest.mark.parametrize(
    "args",
    [
        (sample[:, :1], bins[:1], ranges[:1], arrays[:1], 1),
        (sample[:, :2], bins[:2], ranges[:2], arrays[:2], 2),
        (sample[:, :3], bins[:3], ranges[:3], arrays[:3], 3),
    ],
    ids=lambda x: f"ndim: {x[4]}",
)
def test_from_bins_equals_from_bin_range(
    args: Tuple[np.ndarray, int, tuple, np.ndarray],
):
    """Test whether the numba_histogramdd functions produces the same result
    if called with bin numbers and ranges or with bin edges.

    Args:
        args (Tuple[np.ndarray, int, tuple, np.ndarray, int]):
        Tuple of (samples, bins, ranges, bin_edges, dimension)
    """
    sample_, bins_, ranges_, arrays_, _ = args
    hist1, _ = numba_histogramdd(sample_, bins_, ranges_)
    hist2, _ = numba_histogramdd(sample_, arrays_)
    np.testing.assert_allclose(hist1, hist2)


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
ranges = [[-1, 1], [-2, 2], [-3, 3]]


def test_simplify_binning_arguments_direct():
    """Test the result of the _simplify_binning_arguments functions for number of
    bins and ranges
    """
    bins_, axes_, ranges_ = _simplify_binning_arguments(bins, axes, ranges)
    assert bins_ == bins
    assert axes_ == axes
    assert ranges_ == ranges


def test_simplify_binning_arguments_1d():
    """Test the result of the _simplify_binning_arguments functions for number of
    bins and ranges, 1D case
    """
    bins_, axes_, ranges_ = _simplify_binning_arguments(
        bins[0],
        axes[0],
        ranges[0],
    )
    assert bins_ == [bins[0]]
    assert axes_ == [axes[0]]
    assert ranges_ == ranges[0]


def test_simplify_binning_arguments_edges():
    """Test the result of the _simplify_binning_arguments functions for bin edges"""
    bin_edges = [np.linspace(r[0], r[1], b) for r, b in zip(ranges, bins)]
    bin_edges_, axes_, ranges_ = _simplify_binning_arguments(bin_edges, axes)
    for bin_, bin_edges_ in zip(bin_edges_, bin_edges):
        np.testing.assert_allclose(bin_, bin_edges_)
    assert axes_ == axes
    assert ranges_ is None


def test_simplify_binning_arguments_tuple():
    """Test the result of the _simplify_binning_arguments functions for bin tuples"""
    bin_tuple = [tuple((r[0], r[1], b)) for r, b in zip(ranges, bins)]
    bins_, axes_, ranges_ = _simplify_binning_arguments(bin_tuple, axes)
    assert bins_ == bins
    assert axes_ == axes
    assert ranges_ == ranges

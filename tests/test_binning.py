import numpy as np
import pandas as pd
import pytest

from .helpers import get_linear_bin_edges
from sed.binning import _hist_from_bin_range
from sed.binning import bin_centers_to_bin_edges
from sed.binning import numba_histogramdd


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
def test_histdd_error_is_raised(_samples, _bins):
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
def test_histdd_bins_as_numpy(args):
    sample, bins, _ = args
    H1, _ = np.histogramdd(sample, bins)
    H2, _ = numba_histogramdd(sample, bins)
    return np.testing.assert_allclose(H1, H2)


@pytest.mark.parametrize(
    "args",
    [
        (sample[:, :1], bins[:1], ranges[:1], 1),
        (sample[:, :2], bins[:2], ranges[:2], 2),
        (sample[:, :3], bins[:3], ranges[:3], 3),
    ],
    ids=lambda x: f"ndim: {x[3]}",
)
def test_histdd_ranges_as_numpy(args):
    sample, bins, ranges, _ = args
    H1, _ = np.histogramdd(sample, bins, ranges)
    H2, _ = numba_histogramdd(sample, bins, ranges)
    return np.testing.assert_allclose(H1, H2)


def test_bin_centers_to_bin_edges():
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

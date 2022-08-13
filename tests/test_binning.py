import numpy as np
import pytest

from sed.binning import _hist_from_bin_range
from sed.binning import _hist_from_bins

sample1d = np.random.randn(int(1e2), 1)
sample2d = np.random.randn(int(1e2), 2)
sample3d = np.random.randn(int(1e2), 3)
bins1d = (95,)
bins2d = (95, 34)
bins3d = (95, 34, 27)
ranges1d = np.array([[1, 2]])
ranges2d = np.array([[1, 2], [1, 2]])
ranges3d = np.array([[1, 2], [1, 2], [1, 2]])
arrays1d = [np.linspace(*ranges1d[0], bins1d[0])]
arrays2d = [np.linspace(*ranges2d[i], bins2d[i]) for i in range(2)]
arrays3d = [np.linspace(*ranges3d[i], bins3d[i]) for i in range(3)]


@pytest.mark.parametrize(
    "_samples",
    [sample1d, sample2d, sample3d],
    ids=lambda x: f"samples:{x.shape}",
)
@pytest.mark.parametrize(
    "_bins",
    [bins1d, bins2d, bins3d],
    ids=lambda x: f"bins:{len(x)}",
)
def test_hist_Nd_error_is_raised(_samples, _bins):
    with pytest.raises(ValueError):
        if _samples.shape[1] == len(_bins):
            pytest.skip("Not of interest")
        _hist_from_bin_range(_samples, _bins, ranges1d)


def test_hist_Nd_proper_results():
    H1 = _hist_from_bin_range(sample3d, bins3d, ranges3d)
    H2, _ = np.histogramdd(sample3d, bins3d, ranges3d)
    np.testing.assert_allclose(H1, H2)


@pytest.mark.parametrize(
    "args",
    [
        [sample1d, bins1d, ranges1d, arrays1d],
        [sample2d, bins2d, ranges2d, arrays2d],
        [sample3d, bins3d, ranges3d, arrays3d],
    ],
    ids=lambda x: f"dims: {len(x[1])}",
)
def test_from_bins_equals_from_bin_range(args):
    sample, bins, ranges, arrays = args
    H1 = _hist_from_bin_range(sample, bins, ranges)
    H2 = _hist_from_bins(sample, arrays, tuple(b.size for b in arrays))
    np.testing.assert_allclose(H1, H2)

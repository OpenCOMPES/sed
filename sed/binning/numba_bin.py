from typing import Sequence
from typing import Tuple
from typing import Union

import numba
import numpy as np


@numba.jit(nogil=True, nopython=True)
def _hist_from_bin_range(
    sample: np.array,
    bins: Sequence,
    ranges: np.array,
) -> np.array:
    """
    N dimensional binning function, pre-compiled by Numba for performance.
    Behaves much like numpy.histogramdd, but calculates and returns unsigned 32
    bit integers.

    Args:
        sample: The data to be histogrammed with shape N,D.
        bins: the number of bins for each dimension D.
        range: A sequence of length D, each an optional (lower,
            upper) tuple giving the outer bin edges to be used if the edges are
            not given explicitly in bins.

    Raises:
        ValueError: In case of dimension mismatch.

    Returns:
        The computed histogram.
    """
    ndims = len(bins)
    if sample.shape[1] != ndims:
        raise ValueError(
            "The dimension of bins is not equal to the dimension of the sample x",
        )

    H = np.zeros(bins, np.uint32)
    Hflat = H.ravel()
    delta = np.zeros(ndims, np.float64)
    strides = np.zeros(ndims, np.int64)

    for i in range(ndims):
        delta[i] = 1 / ((ranges[i, 1] - ranges[i, 0]) / bins[i])
        strides[i] = H.strides[i] // H.itemsize

    for t in range(sample.shape[0]):
        is_inside = True
        flatidx = 0
        for i in range(ndims):
            j = (sample[t, i] - ranges[i, 0]) * delta[i]
            is_inside = is_inside and (0 <= j < bins[i])
            flatidx += int(j) * strides[i]
            # don't check all axes if you already know you're out of the range
            if not is_inside:
                break
        if is_inside:
            Hflat[flatidx] += int(is_inside)

    return H


@numba.jit(nogil=True, parallel=False, nopython=True)
def binsearch(bins: np.ndarray, val: float) -> int:
    """Bisection index search function.

    Finds the index of the bin with the highest value below val, i.e. the left edge.
    returns -1 when the value is outside the bin range.

    Args:
        bins: the array on which
        val: value to search for

    Returns:
        int: index of the bin array, returns -1 when value is outside the bins range
    """
    low, high = 0, len(bins) - 1
    mid = high // 2
    if val == bins[high]:
        return high - 1
    if (val < bins[low]) | (val > bins[high]):
        return -1

    while True:
        if val < bins[mid]:
            high = mid
        elif val < bins[mid + 1]:
            return mid
        else:
            low = mid
        mid = (low + high) // 2


@numba.jit(nopython=True, nogil=True, parallel=False)
def _hist_from_bins(
    sample: np.ndarray,
    bins: Sequence[np.ndarray],
    shape: Tuple,
) -> np.ndarray:
    """Numba powered binning method, similar to np.histogramdd.

    Computes the histogram on pre-defined bins.

    Args:
        sample : the array of shape (N,D) on which to compute the histogram
        bins : array of shape (N,D) defining the D bins on which to compute
            the histogram, i.e. the desired output axes.
        shape: shape of the resulting array. Workaround for the fact numba
            does not allow to create tuples.
    Returns:
        hist : the computed n-dimensional histogram
    """
    ndims = len(bins)
    if sample.shape[1] != ndims:
        raise ValueError(
            "The dimension of bins is not equal to the dimension of the sample x",
        )
    H = np.zeros(shape, np.uint32)
    Hflat = H.ravel()

    strides = np.zeros(ndims, np.int64)

    for i in range(ndims):
        strides[i] = H.strides[i] // H.itemsize
    for t in range(sample.shape[0]):
        is_inside = True
        flatidx = 0
        for i in range(ndims):
            j = binsearch(bins[i], sample[t, i])
            # binsearch returns -1 when the value is outside the bin range
            is_inside = is_inside and (j >= 0)
            flatidx += int(j) * strides[i]
            # don't check all axes if you already know you're out of the range
            if not is_inside:
                break
        if is_inside:
            Hflat[flatidx] += int(is_inside)

    return H


def numba_histogramdd(
    sample: np.array,
    bins: Union[Sequence[Union[int, np.ndarray]], np.ndarray],
    ranges: Sequence = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Multidimensional histogramming function, powered by Numba.

    Behaves in total much like numpy.histogramdd. Returns uint32 arrays.
    This was chosen because it has a significant performance improvement over
    uint64 for large binning volumes. Be aware that this can cause overflows
    for very large sample sets exceeding 3E9 counts in a single bin. This
    should never happen in a realistic photoemission experiment with useful bin
    sizes.

    Args:
        sample: The data to be histogrammed with shape N,D
        bins: the number of bins for each dimension D, or a sequence of bins
        on which to calculate the histogram.
        range: The range to use for binning when bins is a list of integers.

    Raises:
        ValueError: In case of dimension mismatch.
        NotImplementedError: When attempting binning in too high number of
        dimensions (>4)
        RuntimeError: Internal shape error after binning

    Returns:
        2-element tuple returned only when returnEdges is True. Otherwise
        only hist is returned.

        - **hist**: The computed histogram
        - **edges**: A list of D arrays describing the bin edges for
            each dimension.
    """

    try:
        # Sample is an ND-array.
        N, D = sample.shape
    except (AttributeError, ValueError):
        # Sample is a sequence of 1D arrays.
        sample = np.atleast_2d(sample).T
        N, D = sample.shape

    if not isinstance(bins, (tuple, list)):
        bins = D * [bins]
    Db = len(bins)
    if isinstance(bins[0], (int, np.int_)):
        method = "int"
    elif isinstance(bins[0], np.ndarray):
        method = "array"
    else:
        raise AttributeError(
            f"bins must be int, np.ndarray or a sequence of the two. "
            f"Found {type(bins[0])} instead",
        )

    if Db != D:  # check number of dimensions
        raise ValueError(
            "The dimension of bins must be equal to the dimension of the sample x.",
        )

    if method == "array":
        hist = _hist_from_bins(
            sample,
            tuple(bins),
            tuple(b.size - 1 for b in bins),
        )
        return hist, bins

    elif method == "int":
        # normalize the range argument
        if ranges is None:
            raise ValueError(
                "must define a value for ranges when bins is"
                " the number of bins",
            )
        #     ranges = (None,) * D
        if D == 1 and isinstance(ranges[0], (int, float)):
            ranges = (ranges,)
        elif len(ranges) != D:
            raise ValueError(
                "range argument must have one entry per dimension",
            )

        ranges = np.asarray(ranges)
        bins = tuple(bins)

        # Create edge arrays
        edges = D * [None]
        nbin = np.empty(D, int)

        for i in range(D):
            edges[i] = np.linspace(*ranges[i, :], bins[i] + 1)

            nbin[i] = len(edges[i]) + 1  # includes an outlier on each end

        hist = _hist_from_bin_range(sample, bins, ranges)

        if (hist.shape != nbin - 2).any():
            raise RuntimeError("Internal Shape Error")

        return hist, edges

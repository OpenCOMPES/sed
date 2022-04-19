# Note: some of the functions presented here were
# inspired by https://github.com/mpes-kit/mpes
from functools import reduce
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import dask.dataframe
import numba
import numpy as np
import pandas as pd
import psutil
import xarray as xr
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

N_CPU = psutil.cpu_count()


def _arraysum(array_a, array_b):
    """Calculate the sum of two arrays."""
    return array_a + array_b


def _simplify_binning_arguments(
    bins: Union[
        int,
        dict,
        tuple,
        List[int],
        List[np.ndarray],
        List[tuple],
    ] = 100,
    axes: Union[str, Sequence[str]] = None,
    ranges: Sequence[Tuple[float, float]] = None,
) -> tuple:
    if isinstance(
        bins,
        dict,
    ):  # bins is a dictionary: unravel to axes and bins
        axes = []
        bins_ = []
        for k, v in bins.items():
            axes.append(k)
            bins_.append(v)
        bins = bins_

    if isinstance(bins, (int, np.ndarray)):
        bins = [bins] * len(axes)
    elif isinstance(bins, tuple):
        if len(bins) == 3:
            bins = [bins]
        else:
            raise ValueError(
                "Bins defined as tuples should only be used to define start "
                / "stop and step of the bins. i.e. should always have lenght 3.",
            )

    assert isinstance(
        bins,
        list,
    ), f"Cannot interpret bins of type {type(bins)}"
    assert axes is not None, "Must define on which axes to bin"
    assert all(isinstance(x, type(bins[0])) for x in bins), (
        "All elements in " "bins must be of the same type"
    )
    # TODO: could implement accepting heterogeneous input.
    bin = bins[0]

    if isinstance(bin, tuple):
        ranges = []
        bins_ = []
        for tpl in bins:
            ranges.append([tpl[0], tpl[1]])
            bins_.append(tpl[2])
        bins = bins_
    elif not isinstance(bin, (int, np.ndarray)):
        raise TypeError(f"Could not interpret bins of type {type(bin)}")

    if ranges is not None:
        if not (len(axes) == len(bins) == len(ranges)):
            raise AttributeError(
                "axes and range and bins must have the same number of elements",
            )
    elif isinstance(bin, int):
        raise AttributeError(
            "Must provide a range if bins is an integer or list of integers",
        )
    elif len(axes) != len(bins):
        raise AttributeError(
            "axes and bins must have the same number of elements",
        )

    return bins, axes, ranges


def bin_partition(
    part: Union[dask.dataframe.core.DataFrame, pd.DataFrame],
    bins: Union[
        int,
        dict,
        tuple,
        List[int],
        List[np.ndarray],
        List[tuple],
    ] = 100,
    axes: Union[str, Sequence[str]] = None,
    ranges: Sequence[Tuple[float, float]] = None,
    histMode: str = "numba",
    jitter: Union[list, dict] = None,
    returnEdges: bool = False,
    skipTest: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, list]]:
    """Compute the n-dimensional histogram of a single dataframe partition.

    Args:
        part: dataframe on which to perform the histogram.
            Usually a partition of a dask DataFrame.
        bins: Definition of the bins. Can  be any of the following cases:
            - an integer describing the number of bins in on all dimensions
            - a tuple of 3 numbers describing start, end and step of the binning range
            - a np.arrays defining the binning edges
            - a list (NOT a tuple) of any of the above (int, tuple or np.ndarray)
            - a dictionary made of the axes as keys and any of the above as values.
            This takes priority over the axes and range arguments.
        axes: The names of the axes (columns) on which to calculate the histogram.
            The order will be the order of the dimensions in the resulting array.
        ranges: list of tuples containing the start and end point of the binning range.
        histMode: Histogram calculation method. Choose between
            "numpy" which uses numpy.histogramdd, and "numba" which uses a
            numba powered similar method.
        jitter: a list of the axes on which to apply jittering.
            To specify the jitter amplitude or method (normal or uniform noise)
            a dictionary can be passed.
            This should look like jitter={'axis':{'amplitude':0.5,'mode':'uniform'}}.
            This example also shows the default behaviour, in case None is
            passed in the dictionary, or jitter is a list of strings.
            Warning: this is not the most performing approach. applying jitter
            on the dataframe before calling the binning is much faster.
        returnEdges: If true, returns a list of D arrays
            describing the bin edges for each dimension, similar to the
            behaviour of np.histogramdd.
        skipTest: turns off input check and data transformation. Defaults to
            False as it is intended for internal use only.
            Warning: setting this True might make error tracking difficult.

    Raises:
        ValueError: When the method requested is not available.
        AttributeError: if bins axes and range are not congruent in dimensionality.
        KeyError: when the columns along which to compute the histogram are not
            present in the dataframe

    Returns:
        2-element tuple returned only when returnEdges is True. Otherwise
            only hist is returned.

            - **hist**: The result of the n-dimensional binning
            - **edges**: A list of D arrays describing the bin edges for
                each dimension.
    """
    if not skipTest:
        bins, axes, ranges = _simplify_binning_arguments(bins, axes, ranges)

    # Locate columns for binning operation
    colID = [part.columns.get_loc(axis) for axis in axes]

    if jitter is not None:
        sel_part = part[axes].copy()

        if isinstance(jitter, Sequence):
            jitter = {k: None for k in jitter}
        for col, jpars in jitter.items():
            if col in axes:
                if jpars is None:
                    jpars = {}
                amp = jpars.get("amplitude", 0.5)
                mode = jpars.get("mode", "uniform")
                axIdx = axes.index(col)
                bin = bins[axIdx]
                if isinstance(bin, int):
                    rng = ranges[axIdx]
                    binsize = abs(rng[1] - rng[0]) / bin
                else:
                    binsize = abs(bin[0] - bin[1])
                    assert np.allclose(
                        binsize,
                        abs(bin[-3] - bin[-2]),
                    ), f"bins along {col} are not uniform. Cannot apply jitter."
                apply_jitter_on_column(sel_part, amp * binsize, col, mode)
        vals = sel_part.values
    else:
        vals = part.values[:, colID]
    if histMode == "numba":
        hist_partition, edges = numba_histogramdd(
            vals,
            bins=bins,
            ranges=ranges,
        )
    elif histMode == "numpy":
        hist_partition, edges = np.histogramdd(
            vals,
            bins=bins,
            range=ranges,
        )
    else:
        raise ValueError(
            f"No binning method {histMode} available. Please choose between "
            f"numba and numpy.",
        )

    if returnEdges:
        return hist_partition, edges
    else:
        return hist_partition


def bin_dataframe(
    df: dask.dataframe.DataFrame,
    bins: Union[
        int,
        dict,
        tuple,
        List[int],
        List[np.ndarray],
        List[tuple],
    ] = 100,
    axes: Union[str, Sequence[str]] = None,
    ranges: Sequence[Tuple[float, float]] = None,
    histMode: str = "numba",
    mode: str = "fast",
    jitter: Union[list, dict] = None,
    pbar: bool = True,
    nCores: int = N_CPU - 1,
    nThreadsPerWorker: int = 4,
    threadpoolAPI: str = "blas",
    **kwds,
) -> xr.DataArray:
    """Computes the n-dimensional histogram on columns of a dataframe,
    parallelized.

    Args:
        df: a dask.DataFrame on which to perform the histogram.
        bins: Definition of the bins. Can  be any of the following cases:
            - an integer describing the number of bins in on all dimensions
            - a tuple of 3 numbers describing start, end and step of the binning range
            - a np.arrays defining the binning edges
            - a list (NOT a tuple) of any of the above (int, tuple or np.ndarray)
            - a dictionary made of the axes as keys and any of the above as values.
            This takes priority over the axes and range arguments.
        axes: The names of the axes (columns) on which to calculate the histogram.
            The order will be the order of the dimensions in the resulting array.
        ranges: list of tuples containing the start and end point of the binning range.
        histMode: Histogram calculation method. Choose between
            "numpy" which uses numpy.histogramdd, and "numba" which uses a
            numba powered similar method.
        mode: Defines how the results from each partition are
            combined.
            Available modes are 'fast', 'lean' and 'legacy'.
        jitter: a list of the axes on which to apply jittering.
            To specify the jitter amplitude or method (normal or uniform noise)
            a dictionary can be passed.
            This should look like jitter={'axis':{'amplitude':0.5,'mode':'uniform'}}.
            This example also shows the default behaviour, in case None is
            passed in the dictionary, or jitter is a list of strings.
        pbar: Allows to deactivate the tqdm progress bar.
        nCores: Number of CPU cores to use for parallelization. Defaults to
            all but one of the available cores.
        nThreadsPerWorker: Limit the number of threads that
            multiprocessing can spawn.
        threadpoolAPI: The API to use for multiprocessing.
        kwds: passed to dask.compute()

    Raises:
        Warning: Warns if there are unimplemented features the user is trying
            to use.
        ValueError: Rises when there is a mismatch in dimensions between the
            binning parameters

    Returns:
        The result of the n-dimensional binning represented in an
            xarray object, combining the data with the axes.
    """

    bins, axes, ranges = _simplify_binning_arguments(bins, axes, ranges)

    # create the coordinate axes for the xarray output
    if isinstance(bins[0], np.ndarray):
        coords = {ax: bin for ax, bin in zip(axes, bins)}
    elif ranges is None:
        raise ValueError(
            "bins is not an array and range is none.. this shouldn't happen.",
        )
    else:
        coords = {
            ax: np.linspace(r[0], r[1], n)
            for ax, r, n in zip(axes, ranges, bins)
        }

    if isinstance(bins[0], np.ndarray):
        fullShape = tuple(x.size for x in bins)
    else:
        fullShape = tuple(bins)

    fullResult = np.zeros(fullShape)
    partitionResults = []  # Partition-level results

    # limit multithreading in worker threads
    with threadpool_limits(limits=nThreadsPerWorker, user_api=threadpoolAPI):

        # Main loop for binning
        for i in tqdm(range(0, df.npartitions, nCores), disable=not (pbar)):

            coreTasks = []  # Core-level jobs
            for j in range(0, nCores):

                ij = i + j
                if ij >= df.npartitions:
                    break

                dfPartition = df.get_partition(
                    ij,
                )  # Obtain dataframe partition
                coreTasks.append(
                    dask.delayed(bin_partition)(
                        dfPartition,
                        bins=bins,
                        axes=axes,
                        ranges=ranges,
                        histMode=histMode,
                        jitter=jitter,
                        skipTest=True,
                        returnEdges=False,
                    ),
                )

            if len(coreTasks) > 0:
                coreResults = dask.compute(*coreTasks, **kwds)

                if mode == "legacy":
                    # Combine all core results for a dataframe partition
                    partitionResult = np.zeros_like(coreResults[0])
                    for coreResult in coreResults:
                        partitionResult += coreResult

                    partitionResults.append(partitionResult)
                    # del partitionResult

                elif mode == "lean":
                    # Combine all core results for a dataframe partition
                    partitionResult = reduce(_arraysum, coreResults)
                    fullResult += partitionResult
                    del partitionResult
                    del coreResults

                elif mode == "fast":
                    combineTasks = []
                    for j in range(0, nCores):
                        combineParts = []
                        # split results along the first dimension among worker
                        # threads
                        for r in coreResults:
                            combineParts.append(
                                r[
                                    int(j * fullShape[0] / nCores) : int(
                                        (j + 1) * fullShape[0] / nCores,
                                    ),
                                    ...,
                                ],
                            )
                        combineTasks.append(
                            dask.delayed(reduce)(_arraysum, combineParts),
                        )
                    combineResults = dask.compute(*combineTasks, **kwds)
                    # Directly fill into target array. This is much faster than
                    # the (not so parallel) reduce/concatenation used before,
                    # and uses less memory.

                    for j in range(0, nCores):
                        fullResult[
                            int(j * fullShape[0] / nCores) : int(
                                (j + 1) * fullShape[0] / nCores,
                            ),
                            ...,
                        ] += combineResults[j]
                    del combineParts
                    del combineTasks
                    del combineResults
                    del coreResults
                else:
                    raise ValueError(f"Could not interpret mode {mode}")

            del coreTasks

    if mode == "legacy":
        # still need to combine all partition results
        fullResult = np.zeros_like(partitionResults[0])
        for pr in partitionResults:
            fullResult += np.nan_to_num(pr)

    da = xr.DataArray(
        data=fullResult.astype("float32"),
        coords=coords,
        dims=list(axes),
    )
    return da


def apply_jitter_on_column(
    df: Union[dask.dataframe.core.DataFrame, pd.DataFrame],
    amp: float,
    col: str,
    mode: str = "uniform",
):
    """Add jittering to the column of a dataframe.

    Args:
        df: Dataframe to add noise/jittering to.
        amp: Amplitude scaling for the jittering noise.
        col: Name of the column to add jittering to.
        mode: Choose between 'uniform' for uniformly
            distributed noise, or 'normal' for noise with normal distribution.
            For columns with digital values, one should choose 'uniform' as
            well as amplitude (amp) equal to the step size.
    """

    colsize = df[col].size
    if mode == "uniform":
        # Uniform Jitter distribution
        df[col] += amp * np.random.uniform(low=-1, high=1, size=colsize)
    elif mode == "normal":
        # Normal Jitter distribution works better for non-linear
        # transformations and jitter sizes that don't match the original bin
        # sizes
        df[col] += amp * np.random.standard_normal(size=colsize)


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

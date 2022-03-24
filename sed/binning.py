# Note: some of the functions presented here were 
# inspired by https://github.com/mpes-kit/mpes
from functools import reduce
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
    """
    Calculate the sum of two arrays.
    """

    return array_a + array_b


def bin_partition(
    part: Union[dask.dataframe.core.DataFrame, pd.DataFrame],
    binDict: dict = None,
    binAxes: Union[str, Sequence[str]] = None,
    binRanges: Sequence[Tuple[float, float]] = None,
    nBins: Union[int, Sequence[int]] = 100,
    hist_mode: str = "numba",
    jitterParams: dict = None,
    return_edges: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, list]]:
    """Compute the n-dimensional histogram of a single dataframe partition.

    Args:
        part: dataframe on which to perform the histogram.
            Usually a partition of a dask DataFrame.
        binDict: TODO: implement passing binning parameters as
            dictionary or other methods
        binAxes: List of names of the axes (columns) on which to
            calculate the histogram.
            The order will be the order of the dimensions in the resulting array.
        nBins: List of number of points along the
            different axes.
        binranges: list of tuples containing the start and
            end point of the binning range.
        hist_mode: Histogram calculation method. Choose between
            "numpy" which uses numpy.histogramdd, and "numba" which uses a
            numba powered similar method.
        jitterParams: Not yet Implemented.
        return_edges: If true, returns a list of D arrays
            describing the bin edges for each dimension, similar to the
            behaviour of np.histogramdd.


    Raises:
        Warning: Warns if there are unimplemented features the user is trying
            to use.
        ValueError: When the method requested is not available.
        KeyError: when the columns along which to compute the histogram are not
            present in the dataframe

    Returns:
        2-element tuple returned only when return_edges is True. Otherwise
            only hist is returned.

            - **hist**: The result of the n-dimensional binning
            - **edges**: A list of D arrays describing the bin edges for
                each dimension.
    """
    if jitterParams is not None:
        raise Warning("Jittering is not yet implemented.")

    cols = part.columns
    # Locate columns for binning operation
    binColumns = [cols.get_loc(binax) for binax in binAxes]

    vals = part.values[:, binColumns]

    if hist_mode == "numba":
        hist_partition, edges = numba_histogramdd(
            vals,
            bins=nBins,
            ranges=binRanges,
        )
    elif hist_mode == "numpy":
        hist_partition, edges = np.histogramdd(
            vals,
            bins=nBins,
            range=binRanges,
        )
    else:
        raise ValueError(
            f"No binning method {hist_mode} available. Please choose between "
            f"numba and numpy.",
        )

    if return_edges:
        return hist_partition, edges
    else:
        return hist_partition


def bin_dataframe(
    df: dask.dataframe.DataFrame,
    binDict: dict = None,
    binAxes: Union[str, Sequence[str]] = None,
    binRanges: Sequence[Tuple[float, float]] = None,
    nBins: Union[int, Sequence[int]] = 100,
    hist_mode: str = "numba",
    mode: str = "fast",
    jitterParams: dict = None,
    pbar: bool = True,
    nCores: int = N_CPU - 1,
    nThreadsPerWorker: int = 4,
    threadpoolAPI: str = "blas",
    **kwds,
) -> xr.DataArray:
    """Computes the n-dimensional histogram on columns of a dataframe,
    parallelized.

    Args:
        df: _description_
        binDict: TODO: implement passing binning parameters as
            dictionary or other methods
        binAxes: List of names of the axes (columns) on which to
            calculate the histogram.
            The order will be the order of the dimensions in the resulting
            array.
        nBins: List of number of points along the
            different axes.
        binranges: list of tuples containing the start and
            end point of the binning range.
        hist_mode: Histogram calculation method. Choose between
            "numpy" which uses numpy.histogramdd, and "numba" which uses a
            numba powered similar method.
        mode: Defines how the results from each partition are
            combined.
            Available modes are 'fast', 'lean' and 'legacy'.
        jitterParams: Not yet Implemented.
        pbar: Allows to deactivate the tqdm progress bar.
        nCores: Number of CPU cores to use for parallelization.
        nThreadsPerWorker: Limit the number of threads that
            multiprocessing can spawn.
        threadpoolAPI: The API to use for multiprocessing.

    Raises:
        Warning: Warns if there are unimplemented features the user is trying
            to use.
        ValueError: Rises when there is a mismatch in dimensions between the
            binning parameters

    Returns:
        The result of the n-dimensional binning represented in an
            xarray object, combining the data with the axes.
    """
    if jitterParams is not None:
        raise Warning("Jittering is not yet implemented.")
    if binDict is not None:
        raise Warning("Usage of binDict is not yet implemented.")

    if isinstance(binAxes, str):
        binAxes = [binAxes]
    elif len(binAxes) != len(binRanges):
        raise ValueError("Must define ranges for all axes")
    elif isinstance(nBins, int):
        nBins = [nBins] * len(binAxes)
    elif len(nBins) != len(binAxes):
        raise ValueError(
            "nBins must be integer or a list of integers for each dimension "
            "in axes.",
        )

    fullResult = np.zeros(tuple(nBins))
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
                        None,
                        binAxes,
                        binRanges,
                        nBins,
                        hist_mode,
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
                                    int(j * nBins[0] / nCores) : int(
                                        (j + 1) * nBins[0] / nCores,
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
                            int(j * nBins[0] / nCores) : int(
                                (j + 1) * nBins[0] / nCores,
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
        coords={
            ax: np.linspace(r[0], r[1], n)
            for ax, r, n in zip(binAxes, binRanges, nBins)
        },
        dims=list(binAxes),
    )
    return da


def applyJitter(
    df: Union[dask.dataframe.core.DataFrame, pd.DataFrame],
    amp: float,
    col: str,
    mode: str = "uniform",
):
    """Add jittering to the column of a dataframe

    Args:
        df: Dataframe to add noise/jittering to.
        amp: Amplitude scaling for the jittering noise.
        col: Name of the column to add jittering to.
        mode: Choose between 'uniform' for uniformly
            distributed noise, or 'normal' for noise with normal distribution.
            For columns with digital values, one should choose 'uniform' as
            well as amplitude (amp) equal to half the step size.
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
def _hist_from_bin_ranges(
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
        ranges: A sequence of length D, each an optional (lower,
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

        if is_inside:
            Hflat[flatidx] += int(is_inside)

    return H


def numba_histogramdd(
    sample: np.array,
    bins: Sequence,
    ranges: Sequence,
) -> Tuple[np.array, np.array]:
    """Wrapper for the Number pre-compiled binning functions.

    Behaves in total much like numpy.histogramdd. Returns uint32 arrays.
    This was chosen because it has a significant performance improvement over
    uint64 for large binning volumes. Be aware that this can cause overflows
    for very large sample sets exceeding 3E9 counts in a single bin. This
    should never happen in a realistic photoemission experiment with useful bin
    sizes.

    Args:
        sample: The data to be histogrammed with shape N,D
        bins: the number of bins for each dimension D
        ranges: the TODO: Missing description

    Raises:
        ValueError: In case of dimension mismatch.
        NotImplementedError: When attempting binning in too high number of
        dimensions (>4)
        RuntimeError: Internal shape error after binning

    Returns:
        2-element tuple returned only when return_edges is True. Otherwise
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

    try:
        M = len(bins)
        if M != D:
            raise ValueError(
                "The dimension of bins must be equal to the dimension of the "
                " sample x.",
            )
    except TypeError:
        # bins is an integer
        bins = D * [bins]

    nbin = np.empty(D, int)
    edges = D * [None]

    # normalize the ranges argument
    if ranges is None:
        ranges = (None,) * D
    elif len(ranges) != D:
        raise ValueError("range argument must have one entry per dimension")

    ranges = np.asarray(ranges)
    bins = tuple(bins)

    # Create edge arrays
    for i in range(D):
        edges[i] = np.linspace(*ranges[i, :], bins[i] + 1)

        nbin[i] = len(edges[i]) + 1  # includes an outlier on each end

    hist = _hist_from_bin_ranges(sample, bins, ranges)

    if (hist.shape != nbin - 2).any():
        raise RuntimeError("Internal Shape Error")

    return hist, edges

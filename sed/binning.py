# All functions in this file are adapted from https://github.com/mpes-kit/mpes
from functools import reduce
from typing import Sequence
from typing import Tuple
from typing import Union

import dask
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
) -> np.ndarray:
    """Compute the n-dimensional histogram of a single dataframe partition.

    Args:
        part (ddf.DataFrame): dataframe on which to perform the histogram.
            Usually a partition of a dask DataFrame.
        binDict (dict, optional): TODO: implement passing binning parameters as
            dictionary or other methods
        binAxes (list): List of names of the axes (columns) on which to
            calculate the histogram.
            The order will be the order of the dimensions in the resulting array.
        nBins (int or list, optional): List of number of points along the
            different axes. Defaults to None.
        binranges (tuple, optional): list of tuples containing the start and
            end point of the binning range. Defaults to None.
        hist_mode (str, optional): Histogram calculation method. Choose between
            "numpy" which uses numpy.histogramdd, and "numba" which uses a
            numba powered similar method. Defaults to 'numba'.
        jitterParams (dict, optional): Not yet Implemented. Defaults to None.
        return_edges: (bool, optional): If true, returns a list of D arrays
            describing the bin edges for each dimension, similar to the
            behaviour of np.histogramdd. Defaults to False


    Raises:
        Warning: Warns if there are unimplemented features the user is trying
            to use.
        ValueError: When the method requested is not available.
        KeyError: when the columns along which to compute the histogram are not
            present in the dataframe

    Returns:
        hist (np.array) : The result of the n-dimensional binning
        edges (list,optional) : A list of D arrays describing the bin edges for
            each dimension.
            This is returned only when return_edges is True.
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
        df (dask.dataframe.DataFrame): _description_
        binDict (dict, optional): TODO: implement passing binning parameters as
            dictionary or other methods
        binAxes (list): List of names of the axes (columns) on which to
            calculate the histogram.
            The order will be the order of the dimensions in the resulting
            array.
        nBins (int or list, optional): List of number of points along the
            different axes. Defaults to None.
        binranges (tuple, optional): list of tuples containing the start and
            end point of the binning range. Defaults to None.
        hist_mode (str, optional): Histogram calculation method. Choose between
            "numpy" which uses numpy.histogramdd, and "numba" which uses a
            numba powered similar method. Defaults to 'numba'.
        mode (str, optional): Defines how the results from each partition are
            combined.
            Available modes are 'fast', 'lean' and 'legacy'. Defaults to 'fast'.
        jitterParams (dict, optional): Not yet Implemented. Defaults to None.
        pbar (bool, optional): Allows to deactivate the tqdm progress bar.
            Defaults to True.
        nCores (int, optional): Number of CPU cores to use for parallelization.
            Defaults to N_CPU-1.
        nThreadsPerWorker (int, optional): Limit the number of threads that
            multiprocessing can spawn. Defaults to 4.
        threadpoolAPI (str, optional): The API to use for multiprocessing.
        Defaults to 'blas'.

    Raises:
        Warning: Warns if there are unimplemented features the user is trying
            to use.
        ValueError: Rises when there is a mismatch in dimensions between the
            binning parameters

    Returns:
        xr.DataArray: The result of the n-dimensional binning represented in an
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
        df (pd.DataFrame): Dataframe to add noise/jittering to.
        amp (float): Amplitude scaling for the jittering noise.
        col (str): Name of the column to add jittering to.
        mode (str, optional): Choose between 'uniform' for uniformly
            distributed noise, or 'normal' for noise with normal distribution.
            For columns with digital values, one should choose 'uniform' as
            well as amplitude (amp) equal to half the step size.
            Defaults to 'uniform'.
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


@numba.jit(nogil=True, parallel=False)
def _hist1d_numba_seq(sample, bins, ranges):
    """
    1D Binning function, pre-compiled by Numba for performance.
    Behaves much like numpy.histogramdd, but calculates and returns unsigned 32
    bit integers
    """
    H = np.zeros((bins[0]), dtype=np.uint32)
    delta = 1 / ((ranges[:, 1] - ranges[:, 0]) / bins)

    if sample.shape[1] != 1:
        raise ValueError(
            "The dimension of bins must be equal to the dimension of the "
            "sample x.",
        )

    for t in range(sample.shape[0]):
        i = (sample[t, 0] - ranges[0, 0]) * delta[0]
        if 0 <= i < bins[0]:
            H[int(i)] += 1

    return H


@numba.jit(nogil=True, parallel=False)
def _hist2d_numba_seq(sample, bins, ranges):
    """
    2D Binning function, pre-compiled by Numba for performance.
    Behaves much like numpy.histogramdd, but calculates and returns unsigned 32
    bit integers
    """
    H = np.zeros((bins[0], bins[1]), dtype=np.uint32)
    delta = 1 / ((ranges[:, 1] - ranges[:, 0]) / bins)

    if sample.shape[1] != 2:
        raise ValueError(
            "The dimension of bins must be equal to the dimension of the "
            "sample x.",
        )

    for t in range(sample.shape[0]):
        i = (sample[t, 0] - ranges[0, 0]) * delta[0]
        j = (sample[t, 1] - ranges[1, 0]) * delta[1]
        if 0 <= i < bins[0] and 0 <= j < bins[1]:
            H[int(i), int(j)] += 1

    return H


@numba.jit(nogil=True, parallel=False)
def _hist3d_numba_seq(sample, bins, ranges):
    """
    3D Binning function, pre-compiled by Numba for performance.
    Behaves much like numpy.histogramdd, but calculates and returns unsigned 32
    bit integers
    """
    H = np.zeros((bins[0], bins[1], bins[2]), dtype=np.uint32)
    delta = 1 / ((ranges[:, 1] - ranges[:, 0]) / bins)

    if sample.shape[1] != 3:
        raise ValueError(
            "The dimension of bins must be equal to the dimension of the "
            "sample x.",
        )

    for t in range(sample.shape[0]):
        i = (sample[t, 0] - ranges[0, 0]) * delta[0]
        j = (sample[t, 1] - ranges[1, 0]) * delta[1]
        k = (sample[t, 2] - ranges[2, 0]) * delta[2]
        if 0 <= i < bins[0] and 0 <= j < bins[1] and 0 <= k < bins[2]:
            H[int(i), int(j), int(k)] += 1

    return H


@numba.jit(nogil=True, parallel=False)
def _hist4d_numba_seq(sample, bins, ranges):
    """
    4D Binning function, pre-compiled by Numba for performance.
    Behaves much like numpy.histogramdd, but calculates and returns unsigned 32
    bit integers
    """
    H = np.zeros((bins[0], bins[1], bins[2], bins[3]), dtype=np.uint32)
    delta = 1 / ((ranges[:, 1] - ranges[:, 0]) / bins)

    if sample.shape[1] != 4:
        raise ValueError(
            "The dimension of bins must be equal to the dimension of the "
            "sample x.",
        )

    for t in range(sample.shape[0]):
        dim_1 = (sample[t, 0] - ranges[0, 0]) * delta[0]
        dim_2 = (sample[t, 1] - ranges[1, 0]) * delta[1]
        dim_3 = (sample[t, 2] - ranges[2, 0]) * delta[2]
        dim_4 = (sample[t, 3] - ranges[3, 0]) * delta[3]
        if (
            0 <= dim_1 < bins[0]
            and 0 <= dim_2 < bins[1]
            and 0 <= dim_3 < bins[2]
            and 0 <= dim_4 < bins[3]
        ):
            H[int(dim_1), int(dim_2), int(dim_3), int(dim_4)] += 1

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
        sample (np.array): The data to be histogrammed with shape N,D
        bins (Sequence): the number of bins for each dimension D
        ranges (Sequence): the

    Raises:
        ValueError: In case of dimension mismatch.
        NotImplementedError: When attempting binning in too high number of
        dimensions (>4)
        RuntimeError: Internal shape error after binning

    Returns:
        hist (np.array): The computed histogram
        edges (np.array): A list of D arrays describing the bin edges for each
        dimension.
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
    bins = np.asarray(bins)

    # Create edge arrays
    for i in range(D):
        edges[i] = np.linspace(*ranges[i, :], bins[i] + 1)

        nbin[i] = len(edges[i]) + 1  # includes an outlier on each end

    if D == 1:
        hist = _hist1d_numba_seq(sample, bins, ranges)
    elif D == 2:
        hist = _hist2d_numba_seq(sample, bins, ranges)
    elif D == 3:
        hist = _hist3d_numba_seq(sample, bins, ranges)
    elif D == 4:
        hist = _hist4d_numba_seq(sample, bins, ranges)
    else:
        raise NotImplementedError(
            "Only implemented for up to 4 dimensions currently.",
        )

    if (hist.shape != nbin - 2).any():
        raise RuntimeError("Internal Shape Error")

    return hist, edges

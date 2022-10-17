from functools import reduce
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import dask.dataframe
import numpy as np
import pandas as pd
import psutil
import xarray as xr
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

from .numba_bin import numba_histogramdd
from .utils import _arraysum
from .utils import _simplify_binning_arguments

N_CPU = psutil.cpu_count()


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
    return_partitions: bool = False,
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
        return_partitions: Option to return a hypercube of dimension n+1,
            where the last dimension corresponds to the dataframe partitions.
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

                if return_partitions:
                    for coreResult in coreResults:
                        partitionResults.append(coreResult)
                    del coreResults

                elif mode == "legacy":
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

    if return_partitions:
        coords = {**coords, **{"df_part": np.arange(df.npartitions)}}
        dims = list(axes)
        dims.append("df_part")
        da = xr.DataArray(
            data=np.stack(partitionResults, axis=-1).astype("float32"),
            coords=coords,
            dims=dims,
        )
        return da

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

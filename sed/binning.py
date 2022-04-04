# All functions in this file are adapted from https://github.com/mpes-kit/mpes
from functools import reduce
from sqlite3 import InternalError
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


def _array_sum(array_a, array_b):
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
                "Bins defined as tuples should only be used to define start stop and step of the bins. i.e. should always have lenght 3.",
            )

    assert isinstance(
        bins,
        list,
    ), f"Cannot interpret bins of type {type(bins)}"
    assert axes is not None, f"Must define on which axes to bin"
    assert all(
        type(x) == type(bins[0]) for x in bins
    ), "All elements in bins must be of the same type"
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
    hist_mode: str = "numba",
    jitter: Union[list, dict] = None,
    return_edges: bool = False,
    skip_test: bool = False,
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
        his_mMode: Histogram calculation method. Choose between
            "numpy" which uses numpy.histogramdd, and "numba" which uses a
            numba powered similar method.
        jitter: a list of the axes on which to apply jittering.
            To specify the jitter amplitude or method (normal or uniform noise) a dictionary can be passed.
            This should look like jitter={'axis':{'amplitude':0.5,'mode':'uniform'}}.
            This example also shows the default behaviour, in case None is passed in the dictionary, or jitter is a list of strings.
            Warning: this is not the most performing approach. applying jitter on the dataframe before calling the binning is much faster.
        return_edges: If true, returns a list of D arrays
            describing the bin edges for each dimension, similar to the
            behaviour of np.histogramdd.
        skipTest: turns off input check and data transformation. Defaults to False as it is intended for internal use only. Warning: setting this True might make error tracking difficult.

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
    if not skip_test:
        bins, axes, ranges = _simplify_binning_arguments(bins, axes, ranges)

    # Locate columns for binning operation
    col_ID = [part.columns.get_loc(axis) for axis in axes]

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
                ax_idx = axes.index(col)
                bin = bins[ax_idx]
                if isinstance(bin, int):
                    rng = ranges[ax_idx]
                    bin_size = abs(rng[1] - rng[0]) / bin
                else:
                    bin_size = abs(bin[0] - bin[1])
                    assert np.allclose(
                        bin_size,
                        abs(bin[-3] - bin[-2]),
                    ), f"bins along {col} are not uniform. Cannot apply jitter."
                apply_jitter_on_column(sel_part, amp * bin_size, col, mode)
        vals = sel_part.values
    else:
        vals = part.values[:, col_ID]
    if hist_mode == "numba":
        hist_partition, edges = numba_histogramdd(
            vals,
            bins=bins,
            ranges=ranges,
        )
    elif hist_mode == "numpy":
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

    if return_edges:
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
    hist_mode: str = "numba",
    mode: str = "fast",
    jitter: Union[list, dict] = None,
    pbar: bool = True,
    n_cores: int = N_CPU - 1,
    n_threads_per_worker: int = 4,
    threadpool_API: str = "blas",
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
        hist_mMode: Histogram calculation method. Choose between
            "numpy" which uses numpy.histogramdd, and "numba" which uses a
            numba powered similar method.
        mode: Defines how the results from each partition are
            combined.
            Available modes are 'fast', 'lean' and 'legacy'.
        jitter: a list of the axes on which to apply jittering.
            To specify the jitter amplitude or method (normal or uniform noise) a dictionary can be passed.
            This should look like jitter={'axis':{'amplitude':0.5,'mode':'uniform'}}.
            This example also shows the default behaviour, in case None is passed in the dictionary, or jitter is a list of strings.
        pbar: Allows to deactivate the tqdm progress bar.
        n_cores: Number of CPU cores to use for parallelization. Defaults to all but one of the available cores.
        n_threads_per_worker: Limit the number of threads that
            multiprocessing can spawn.
        threadpool_API: The API to use for multiprocessing.
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
        full_shape = tuple(x.size for x in bins)
    else:
        full_shape = tuple(bins)

    full_result = np.zeros(full_shape)
    partition_results = []  # Partition-level results

    # limit multithreading in worker threads
    with threadpool_limits(limits=n_threads_per_worker, user_api=threadpool_API):

        # Main loop for binning
        for i in tqdm(range(0, df.n_partitions, n_cores), disable=not (pbar)):

            core_tasks = []  # Core-level jobs
            for j in range(0, n_cores):

                ij = i + j
                if ij >= df.n_partitions:
                    break

                df_partition = df.get_partition(
                    ij,
                )  # Obtain dataframe partition
                core_tasks.append(
                    dask.delayed(bin_partition)(
                        df_partition,
                        bins=bins,
                        axes=axes,
                        ranges=ranges,
                        hist_mode=hist_mode,
                        jitter=jitter,
                        skip_test=True,
                        return_edges=False,
                    ),
                )

            if len(core_tasks) > 0:
                core_results = dask.compute(*core_tasks, **kwds)

                if mode == "legacy":
                    # Combine all core results for a dataframe partition
                    partition_result = np.zeros_like(core_results[0])
                    for core_result in core_results:
                        partition_result += core_result

                    partition_results.append(partition_result)
                    # del partition_result

                elif mode == "lean":
                    # Combine all core results for a dataframe partition
                    partition_result = reduce(_array_sum, core_results)
                    full_result += partition_result
                    del partition_result
                    del core_results

                elif mode == "fast":
                    combine_tasks = []
                    for j in range(0, n_cores):
                        combine_parts = []
                        # split results along the first dimension among worker
                        # threads
                        for r in core_results:
                            combine_parts.append(
                                r[
                                    int(j * full_shape[0] / n_cores) : int(
                                        (j + 1) * full_shape[0] / n_cores,
                                    ),
                                    ...,
                                ],
                            )
                        combine_tasks.append(
                            dask.delayed(reduce)(_array_sum, combine_parts),
                        )
                    combine_results = dask.compute(*combine_tasks, **kwds)
                    # Directly fill into target array. This is much faster than
                    # the (not so parallel) reduce/concatenation used before,
                    # and uses less memory.

                    for j in range(0, n_cores):
                        full_result[
                            int(j * full_shape[0] / n_cores) : int(
                                (j + 1) * full_shape[0] / n_cores,
                            ),
                            ...,
                        ] += combine_results[j]
                    del combine_parts
                    del combine_tasks
                    del combine_results
                    del core_results
                else:
                    raise ValueError(f"Could not interpret mode {mode}")

            del core_tasks

    if mode == "legacy":
        # still need to combine all partition results
        full_result = np.zeros_like(partition_results[0])
        for pr in partition_results:
            full_result += np.nan_to_num(pr)

    da = xr.DataArray(
        data=full_result.astype("float32"),
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

    col_size = df[col].size
    if mode == "uniform":
        # Uniform Jitter distribution
        df[col] += amp * np.random.uniform(low=-1, high=1, size=col_size)
    elif mode == "normal":
        # Normal Jitter distribution works better for non-linear
        # transformations and jitter sizes that don't match the original bin
        # sizes
        df[col] += amp * np.random.standard_normal(size=col_size)


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
    n_dims = len(bins)
    if sample.shape[1] != n_dims:
        raise ValueError(
            "The dimension of bins is not equal to the dimension of the sample x",
        )

    H = np.zeros(bins, np.uint32)
    H_flat = H.ravel()
    delta = np.zeros(n_dims, np.float64)
    strides = np.zeros(n_dims, np.int64)

    for i in range(n_dims):
        delta[i] = 1 / ((ranges[i, 1] - ranges[i, 0]) / bins[i])
        strides[i] = H.strides[i] // H.itemsize

    for t in range(sample.shape[0]):
        is_inside = True
        flat_idx = 0
        for i in range(ndims):
            j = (sample[t, i] - ranges[i, 0]) * delta[i]
            is_inside = is_inside and (0 <= j < bins[i])
            flat_idx += int(j) * strides[i]
            # don't check all axes if you already know you're out of the range
            if not is_inside:
                break
        if is_inside:
            H_flat[flat_idx] += int(is_inside)

    return H


@numba.jit(nogil=True, parallel=False, nopython=True)
def bin_search(bins: np.ndarray, val: float) -> int:
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
    if (val < bins[low]) | (val >= bins[high]):
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
        bins : array of shape (N,D) defining the D bins on which to compute the histogram, i.e. the desired output axes
        shape: shape of the resulting array. Workaround for the fact numba does not allow to create tuples
    Returns:
        hist : the computed n-dimensional histogram
    """
    n_dims = len(bins)
    if sample.shape[1] != n_dims:
        raise ValueError(
            "The dimension of bins is not equal to the dimension of the sample x",
        )
    H = np.zeros(shape, np.uint32)
    H_flat = H.ravel()

    strides = np.zeros(n_dims, np.int64)

    for i in range(n_dims):
        strides[i] = H.strides[i] // H.itemsize
    for t in range(sample.shape[0]):
        is_inside = True
        flat_idx = 0
        for i in range(n_dims):
            j = bin_search(bins[i], sample[t, i])
            # bin_search returns -1 when the value is outside the bin range
            is_inside = is_inside and (j >= 0)
            flat_idx += int(j) * strides[i]
            # don't check all axes if you already know you're out of the range
            if not is_inside:
                break
        if is_inside:
            H_flat[flat_idx] += int(is_inside)

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
        bins: the number of bins for each dimension D, or a sequence of bins on which to calculate the histogram.
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

    if isinstance(bins, int):
        bins = D * [bins]
        method = "int"
        Db = len(bins)
    try:
        Db = len(bins)
        if isinstance(bins[0], int):
            method = "int"
        elif isinstance(bins[0], np.ndarray):
            method = "array"
    except AttributeError:
        # bins is a single integer
        bins = D * [bins]
        method = "int"
        Db = len(bins)

    if Db != D:  # check number of dimensions
        raise ValueError(
            "The dimension of bins must be equal to the dimension of the sample x.",
        )

    if method == "array":
        hist = _hist_from_bins(
            sample,
            tuple(bins),
            tuple(b.size for b in bins),
        )
        return hist, bins

    elif method == "int":
        # normalize the range argument
        if ranges is None:
            ranges = (None,) * D
        elif len(ranges) != D:
            raise ValueError(
                "range argument must have one entry per dimension",
            )

        ranges = np.asarray(ranges)
        bins = tuple(bins)

        # Create edge arrays
        edges = D * [None]
        n_bin = np.empty(D, int)

        for i in range(D):
            edges[i] = np.linspace(*ranges[i, :], bins[i] + 1)

            n_bin[i] = len(edges[i]) + 1  # includes an outlier on each end

        hist = _hist_from_bin_range(sample, bins, ranges)

        if (hist.shape != n_bin - 2).any():
            raise RuntimeError("Internal Shape Error")

        return hist, edges

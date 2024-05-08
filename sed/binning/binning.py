"""This module contains the binning functions of the sed.binning module

"""
import gc
from functools import reduce
from typing import cast
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
from .utils import bin_centers_to_bin_edges
from .utils import simplify_binning_arguments

N_CPU = psutil.cpu_count()


def bin_partition(
    part: Union[dask.dataframe.DataFrame, pd.DataFrame],
    bins: Union[
        int,
        dict,
        Sequence[int],
        Sequence[np.ndarray],
        Sequence[tuple],
    ] = 100,
    axes: Sequence[str] = None,
    ranges: Sequence[Tuple[float, float]] = None,
    hist_mode: str = "numba",
    jitter: Union[list, dict] = None,
    return_edges: bool = False,
    skip_test: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, list]]:
    """Compute the n-dimensional histogram of a single dataframe partition.

    Args:
        part (Union[dask.dataframe.DataFrame, pd.DataFrame]): dataframe on which
            to perform the histogram. Usually a partition of a dask DataFrame.
        bins (int, dict, Sequence[int], Sequence[np.ndarray], Sequence[tuple], optional):
            Definition of the bins. Can  be any of the following cases:

                - an integer describing the number of bins for all dimensions. This
                  requires "ranges" to be defined as well.
                - A sequence containing one entry of the following types for each
                  dimenstion:

                    - an integer describing the number of bins. This requires "ranges"
                      to be defined as well.
                    - a np.arrays defining the bin centers
                    - a tuple of 3 numbers describing start, end and step of the binning
                      range.

                - a dictionary made of the axes as keys and any of the above as
                  values.

            The last option takes priority over the axes and range arguments.
            Defaults to 100.
        axes (Sequence[str], optional): Sequence containing the names of
            the axes (columns) on which to calculate the histogram. The order will be
            the order of the dimensions in the resulting array. Only not required if
            bins are provided as dictionary containing the axis names.
            Defaults to None.
        ranges (Sequence[Tuple[float, float]], optional): Sequence of tuples containing
            the start and end point of the binning range. Required if bins given as
            int or Sequence[int]. Defaults to None.
        hist_mode (str, optional): Histogram calculation method.

                - "numpy": use ``numpy.histogramdd``,
                - "numba" use a numba powered similar method.

            Defaults to "numba".
        jitter (Union[list, dict], optional): a list of the axes on which to apply
            jittering. To specify the jitter amplitude or method (normal or uniform
            noise) a dictionary can be passed. This should look like
            jitter={'axis':{'amplitude':0.5,'mode':'uniform'}}.
            This example also shows the default behaviour, in case None is
            passed in the dictionary, or jitter is a list of strings.
            Warning: this is not the most performing approach. Applying jitter
            on the dataframe before calling the binning is much faster.
            Defaults to None.
        return_edges (bool, optional): If True, returns a list of D arrays
            describing the bin edges for each dimension, similar to the
            behaviour of ``np.histogramdd``. Defaults to False.
        skip_test (bool, optional): Turns off input check and data transformation.
            Defaults to False as it is intended for internal use only.
            Warning: setting this True might make error tracking difficult.

    Raises:
        ValueError: When the method requested is not available.
        AttributeError: if bins axes and range are not congruent in dimensionality.
        KeyError: when the columns along which to compute the histogram are not
            present in the dataframe

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, list]]: 2-element tuple returned only when
        returnEdges is True. Otherwise only hist is returned.

        - **hist**: The result of the n-dimensional binning
        - **edges**: A list of D arrays describing the bin edges for each dimension.
    """
    if not skip_test:
        bins, axes, ranges = simplify_binning_arguments(bins, axes, ranges)
    else:
        if not isinstance(bins, list) or not (
            all(isinstance(x, (int, np.int64)) for x in bins)
            or all(isinstance(x, np.ndarray) for x in bins)
        ):
            raise TypeError(
                "bins needs to be of type 'List[int] or List[np.ndarray]' if tests are skipped!",
            )
        if not (isinstance(axes, list)) or not all(isinstance(axis, str) for axis in axes):
            raise TypeError(
                "axes needs to be of type 'List[str]' if tests are skipped!",
            )
        bins = cast(Union[List[int], List[np.ndarray]], bins)
        axes = cast(List[str], axes)
        ranges = cast(List[Tuple[float, float]], ranges)

    # convert bin centers to bin edges:
    if all(isinstance(x, np.ndarray) for x in bins):
        bins = cast(List[np.ndarray], bins)
        for i, bin_centers in enumerate(bins):
            bins[i] = bin_centers_to_bin_edges(bin_centers)
    else:
        bins = cast(List[int], bins)
        # shift ranges by half a bin size to align the bin centers to the given ranges,
        # as the histogram functions interprete the ranges as limits for the edges.
        for i, nbins in enumerate(bins):
            halfbinsize = (ranges[i][1] - ranges[i][0]) / (nbins) / 2
            ranges[i] = (
                ranges[i][0] - halfbinsize,
                ranges[i][1] - halfbinsize,
            )

    # Locate columns for binning operation
    col_id = [part.columns.get_loc(axis) for axis in axes]

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
                ax_index = axes.index(col)
                _bin = bins[ax_index]
                if isinstance(_bin, (int, np.int64)):
                    rng = ranges[ax_index]
                    binsize = abs(rng[1] - rng[0]) / _bin
                else:
                    binsize = abs(_bin[0] - _bin[1])
                    assert np.allclose(
                        binsize,
                        abs(_bin[-3] - _bin[-2]),
                    ), f"bins along {col} are not uniform. Cannot apply jitter."
                apply_jitter_on_column(sel_part, amp * binsize, col, mode)
        vals = sel_part.values
    else:
        vals = part.iloc[:, col_id].values
    if vals.dtype == "object":
        raise ValueError(
            "Binning requires all binned dataframe columns to be of numeric type. "
            "Encountered data types were "
            f"{[part.columns[id] + ': ' + str(part.iloc[:, id].dtype) for id in col_id]}. "
            "Please make sure all axes data are of numeric type.",
        )
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
            f"No binning method {hist_mode} available. Please choose between " f"numba and numpy.",
        )

    if return_edges:
        return hist_partition, edges

    return hist_partition


def bin_dataframe(
    df: dask.dataframe.DataFrame,
    bins: Union[
        int,
        dict,
        Sequence[int],
        Sequence[np.ndarray],
        Sequence[tuple],
    ] = 100,
    axes: Sequence[str] = None,
    ranges: Sequence[Tuple[float, float]] = None,
    hist_mode: str = "numba",
    mode: str = "fast",
    jitter: Union[list, dict] = None,
    pbar: bool = True,
    n_cores: int = N_CPU - 1,
    threads_per_worker: int = 4,
    threadpool_api: str = "blas",
    return_partitions: bool = False,
    **kwds,
) -> xr.DataArray:
    """Computes the n-dimensional histogram on columns of a dataframe,
    parallelized.

    Args:
        df (dask.dataframe.DataFrame): a dask.DataFrame on which to perform the
            histogram.
            bins (int, dict, Sequence[int], Sequence[np.ndarray], Sequence[tuple], optional):
            Definition of the bins. Can be any of the following cases:

                - an integer describing the number of bins for all dimensions. This
                  requires "ranges" to be defined as well.
                - A sequence containing one entry of the following types for each
                  dimenstion:

                    - an integer describing the number of bins. This requires "ranges"
                      to be defined as well.
                    - a np.arrays defining the bin centers
                    - a tuple of 3 numbers describing start, end and step of the binning
                      range.

                - a dictionary made of the axes as keys and any of the above as
                  values.

            The last option takes priority over the axes and range arguments.
            Defaults to 100.
        axes (Sequence[str], optional): Sequence containing the names of
            the axes (columns) on which to calculate the histogram. The order will be
            the order of the dimensions in the resulting array. Only not required if
            bins are provided as dictionary containing the axis names.
            Defaults to None.
        ranges (Sequence[Tuple[float, float]], optional): Sequence of tuples containing
            the start and end point of the binning range. Required if bins given as
            int or Sequence[int]. Defaults to None.
        hist_mode (str, optional): Histogram calculation method.

                - "numpy": use ``numpy.histogramdd``,
                - "numba" use a numba powered similar method.

            Defaults to "numba".
        mode (str, optional): Defines how the results from each partition are combined.

                - 'fast': Uses parallelized recombination of results.
                - 'lean': Store all partition results in a list, and recombine at the
                  end.
                - 'legacy': Single-core recombination of partition results.

            Defaults to "fast".
        jitter (Union[list, dict], optional): a list of the axes on which to apply
            jittering. To specify the jitter amplitude or method (normal or uniform
            noise) a dictionary can be passed. This should look like
            jitter={'axis':{'amplitude':0.5,'mode':'uniform'}}.
            This example also shows the default behaviour, in case None is
            passed in the dictionary, or jitter is a list of strings.
            Warning: this is not the most performing approach. applying jitter
            on the dataframe before calling the binning is much faster.
            Defaults to None.
        pbar (bool, optional): Option to show the tqdm progress bar. Defaults to True.
        n_cores (int, optional): Number of CPU cores to use for parallelization.
            Defaults to all but one of the available cores. Defaults to N_CPU-1.
        threads_per_worker (int, optional): Limit the number of threads that
            multiprocessing can spawn. Defaults to 4.
        threadpool_api (str, optional): The API to use for multiprocessing.
            Defaults to "blas".
        return_partitions (bool, optional): Option to return a hypercube of dimension
            n+1, where the last dimension corresponds to the dataframe partitions.
            Defaults to False.
        **kwds: Keyword arguments passed to ``dask.compute()``

    Raises:
        Warning: Warns if there are unimplemented features the user is trying to use.
        ValueError: Raised when there is a mismatch in dimensions between the
            binning parameters.

    Returns:
        xr.DataArray: The result of the n-dimensional binning represented in an
        xarray object, combining the data with the axes (bin centers).
    """
    bins, axes, ranges = simplify_binning_arguments(bins, axes, ranges)

    # create the coordinate axes for the xarray output
    # if provided as array, they are interpreted as bin centers
    if isinstance(bins[0], np.ndarray):
        bins = cast(List[np.ndarray], bins)
        coords = dict(zip(axes, bins))
    elif ranges is None:
        raise ValueError(
            "bins is not an array and range is none. this shouldn't happen.",
        )
    else:
        bins = cast(List[int], bins)
        coords = {
            ax: np.linspace(r[0], r[1], n, endpoint=False) for ax, r, n in zip(axes, ranges, bins)
        }

    full_shape = tuple(axis.size for axis in coords.values())

    full_result = np.zeros(full_shape)
    partition_results = []  # Partition-level results

    # limit multithreading in worker threads
    with threadpool_limits(limits=threads_per_worker, user_api=threadpool_api):
        # Main loop for binning
        for i in tqdm(range(0, df.npartitions, n_cores), disable=not pbar):
            core_tasks = []  # Core-level jobs
            for j in range(0, n_cores):
                partition_index = i + j
                if partition_index >= df.npartitions:
                    break

                df_partition = df.get_partition(
                    partition_index,
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

                if return_partitions:
                    for core_result in core_results:
                        partition_results.append(core_result)
                    del core_results

                elif mode == "legacy":
                    # Combine all core results for a dataframe partition
                    partition_result = np.zeros_like(core_results[0])
                    for core_result in core_results:
                        partition_result += core_result

                    partition_results.append(partition_result)
                    # del partitionResult

                elif mode == "lean":
                    # Combine all core results for a dataframe partition
                    partition_result = reduce(_arraysum, core_results)
                    full_result += partition_result
                    del partition_result
                    del core_results

                elif mode == "fast":
                    combine_tasks = []
                    for j in range(0, n_cores):
                        combine_parts = []
                        # split results along the first dimension among worker
                        # threads
                        for core_result in core_results:
                            combine_parts.append(
                                core_result[
                                    int(j * full_shape[0] / n_cores) : int(
                                        (j + 1) * full_shape[0] / n_cores,
                                    ),
                                    ...,
                                ],
                            )
                        combine_tasks.append(
                            dask.delayed(reduce)(_arraysum, combine_parts),
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

    if return_partitions:
        coords = {**coords, **{"df_part": np.arange(df.npartitions)}}
        dims = list(axes)
        dims.append("df_part")
        data_array = xr.DataArray(
            data=np.stack(partition_results, axis=-1).astype("float32"),
            coords=coords,
            dims=dims,
        )

    else:
        if mode == "legacy":
            # still need to combine all partition results
            full_result = np.zeros_like(partition_results[0])
            for partition_result in partition_results:
                full_result += np.nan_to_num(partition_result)

        data_array = xr.DataArray(
            data=full_result.astype("float32"),
            coords=coords,
            dims=list(axes),
        )

    gc.collect()
    return data_array


def normalization_histogram_from_timestamps(
    df: dask.dataframe.DataFrame,
    axis: str,
    bin_centers: np.ndarray,
    time_stamp_column: str,
) -> xr.DataArray:
    """Get a normalization histogram from the time stamps column in the dataframe.

    Args:
        df (dask.dataframe.DataFrame): a dask.DataFrame on which to perform the
            histogram.
        axis (str): The axis (dataframe column) on which to calculate the normalization
            histogram.
        bin_centers (np.ndarray): Bin centers used for binning of the axis.
        time_stamp_column (str): Dataframe column containing the time stamps.

    Returns:
        xr.DataArray: Calculated normalization histogram.
    """
    time_per_electron = df[time_stamp_column].diff()

    bins = df[axis].map_partitions(
        pd.cut,
        bins=bin_centers_to_bin_edges(bin_centers),
    )

    histogram = time_per_electron.groupby([bins]).sum().compute().values

    data_array = xr.DataArray(
        data=histogram,
        coords={axis: bin_centers},
    )

    return data_array


def normalization_histogram_from_timed_dataframe(
    df: dask.dataframe.DataFrame,
    axis: str,
    bin_centers: np.ndarray,
    time_unit: float,
) -> xr.DataArray:
    """Get a normalization histogram from a timed datafram.

    Args:
        df (dask.dataframe.DataFrame): a dask.DataFrame on which to perform the
            histogram. Entries should be based on an equal time unit.
        axis (str): The axis (dataframe column) on which to calculate the normalization
            histogram.
        bin_centers (np.ndarray): Bin centers used for binning of the axis.
        time_unit (float): Time unit the data frame entries are based on.

    Returns:
        xr.DataArray: Calculated normalization histogram.
    """
    bins = df[axis].map_partitions(
        pd.cut,
        bins=bin_centers_to_bin_edges(bin_centers),
    )

    histogram = df[axis].groupby([bins]).count().compute().values * time_unit
    # histogram = bin_dataframe(df, axes=[axis], bins=[bin_centers]) * time_unit

    data_array = xr.DataArray(
        data=histogram,
        coords={axis: bin_centers},
    )

    return data_array


def apply_jitter_on_column(
    df: Union[dask.dataframe.core.DataFrame, pd.DataFrame],
    amp: float,
    col: str,
    mode: str = "uniform",
):
    """Add jittering to the column of a dataframe.

    Args:
        df (Union[dask.dataframe.core.DataFrame, pd.DataFrame]): Dataframe to add
            noise/jittering to.
        amp (float): Amplitude scaling for the jittering noise.
        col (str): Name of the column to add jittering to.
        mode (str, optional): Choose between 'uniform' for uniformly
            distributed noise, or 'normal' for noise with normal distribution.
            For columns with digital values, one should choose 'uniform' as
            well as amplitude (amp) equal to the step size. Defaults to "uniform".
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

"""
Binning functions for the toolbox_sxp package.

Inspired by the SED library:
<https://github.com/OpenCOMPES/sed>

Author: David Doblas Jiménez <david.doblas-jimenez@xfel.eu>
Copyright (c) 2023, European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.

You should have received a copy of the 3-Clause BSD License along with this
program. If not, see <https://opensource.org/licenses/BSD-3-Clause>
"""
import multiprocessing as mp

import h5py
import numba
import numpy as np
from scipy.signal import savgol_filter

from .reader import raw_hdf


@numba.jit(nogil=True, nopython=True)
def _hist_from_bin_range(sample, bins, ranges):
    """Get a histogram using specific ranges

    Compute the histogram of an array given the ranges and bins to be used.

    Args:
        sample: numpy.ndarray
            Input data
        bins: tuple
            Bin numbers to use for binning
        ranges: tuple
            Ranges to use for binning

    Returns:
        (numpy.ndarray, list)
            An array with the values of the histogram and a list with the bin
            edges (length(array)+1)
    """
    ndims = len(bins)
    if sample.shape[1] != ndims:
        raise ValueError(
            "The dimension of bins is not equal to the dimension of the sample",
        )

    H = np.zeros(bins, np.float64)
    Hflat = H.ravel()
    delta = np.zeros(ndims, np.float64)
    strides = np.zeros(ndims, np.int64)
    edges = []

    for i in range(ndims):
        delta[i] = 1 / ((ranges[i, 1] - ranges[i, 0]) / bins[i])
        strides[i] = H.strides[i] // H.itemsize
        edges.append(np.linspace(ranges[i][0], ranges[i][1], bins[i] + 1))

    for t in range(sample.shape[0]):
        is_inside = True
        flatidx = 0
        for i in range(ndims):
            j = (sample[t, i] - ranges[i, 0]) * delta[i]
            j = j - 0.5 * (j == bins[i])
            is_inside = is_inside and (0 <= j < bins[i])
            flatidx += int(j) * strides[i]
            # don't check all axes if you already know you're out of the range
            if not is_inside:
                break
        if is_inside:
            Hflat[flatidx] += int(is_inside)

    return H, edges


def _binning_files(
    files,
    axes=None,
    bins=None,
    ranges=None,
    jittered=False,
    jitter_amp=0.5,
    axis_min=None,
    axis_max=None,
):
    """

    Args:
        files: str
            Path of the directory where the files are located. Accepts also a
            Path object
        axes: tuple
            Axes to bin
        bins: tuple
            Bin numbers to use for binning
        ranges: tuple
            Ranges to use for binning
        jittered: bool
            False (default) will not apply jittering, i.e., a uniformly
            distribution to be added to the raw data
        jitter_amp: float
            Factor for multiplying the jitter. Default to 0.5
        axis_min: int
            None (default) will start from the 0th index in the `axes` columns
            from hdf5 files
        axis_max: int
            None (default) will take up to the last index in the `axes` columns
            from hdf5 files

    Returns:

    """
    hists = []
    counts = 0

    for filename in files:
        with h5py.File(filename, "r") as hf:
            data = {
                ax: hf[raw_hdf[ax]][axis_min:axis_max].astype(np.float32)
                for ax in axes
            }

        if jittered:
            rng = np.random.default_rng(seed=0)
            sz = data[axes[0]].size
            for jax, jb, jr in zip(axes, bins, ranges):
                # Calculate the bar size of the histogram in every dimension
                binsz = abs(jr[0] - jr[1]) / jb
                # Jitter as random uniformly distributed noise (W. S. Cleveland)
                data[jax] += (
                    2
                    * jitter_amp
                    * binsz
                    * (rng.uniform(low=-0.5, high=0.5, size=sz))
                )

        data_nb, edges_nb = _hist_from_bin_range(
            sample=np.stack((data[a] for a in axes), axis=-1),
            bins=bins,
            ranges=np.asarray([r for r in ranges]),
        )
        counts = counts + data[next(iter(data))].size
        hists.append(data_nb)

    v = dict()
    histcoord = "midpoint"
    if histcoord == "midpoint":
        v["axes"] = {
            axis: ((edges_nb[idx][1:] + edges_nb[idx][:-1]) / 2.0).astype(
                np.float32,
            )
            for idx, axis in enumerate(axes)
        }
    elif histcoord == "edge":
        v["axes"] = dict(zip(axes, edges_nb))
    else:
        raise ValueError("Unexpected value in 'histcoord'")

    v["binned"] = np.sum(np.array(hists), axis=0)
    v["counts"] = counts

    return v


def pbinning(
    files,
    nproc=10,
    axes=None,
    bins=None,
    ranges=None,
    jittered=False,
    jitter_amp=0.5,
    axis_min=None,
    axis_max=None,
    sum_binned=True,
    normalize=False,
    smooth=False,
):
    """

    Args:
        files: str
            Path of the directory where the files are located. Accepts also a
            Path object
        nproc: int
            Number of workers to be used in the multiprocessing pool
        axes: tuple
            Axes to bin
        bins: tuple
            Bin numbers to use for binning
        sliceranges: tuple
            Ranges to use for binning
        jittered: bool
            False (default) will not apply jittering, i.e., a uniformly
            distribution to be added to the raw data
        jitter_amp: float
            Factor for multiplying the jitter. Default to 0.5
        axis_min: int
            None (default) will start from the 0th index in the `axes` columns
            from hdf5 files
        axis_max: int
            None (default) will take up to the last index in the `axes` columns
            from hdf5 files
        sum_binned: bool
            True (default) will return a sum of the `binned` array
        normalize: bool
            False (default) will not normalize the `binned` arrays. Incompatible
            with the `sum_binned` flag
        smooth: bool
            False (default) will not apply a `Savitzky-Golay` filter to the
            normalized `binned` arrays. To be used with the `normalize` flag

    Returns:
        dict
            Dictionary with `axes`, `binned` and `counts`

    """
    assert nproc <= len(
        files,
    ), f"Can not use {nproc} process to compute {len(files)} files"

    chunks = [files[i::nproc] for i in range(nproc)]

    with mp.Pool(nproc) as pool:
        res = pool.starmap(
            _binning_files,
            zip(
                chunks,
                [axes] * len(chunks),
                [bins] * len(chunks),
                [ranges] * len(chunks),
                [jittered] * len(chunks),
                [jitter_amp] * len(chunks),
                [axis_min] * len(chunks),
                [axis_max] * len(chunks),
            ),
        )

    binned = []
    counts = 0
    for chunk in res:
        binned.append(chunk["binned"])
        counts = counts + chunk["counts"]

    v = res[0].copy()
    if sum_binned:
        v["binned"] = np.sum(np.array(binned), axis=0).astype(np.float32)
    else:
        v["binned"] = np.array(binned)

    if normalize and not sum_binned:
        if smooth:
            v["binned"] = savgol_filter(
                v["binned"],
                window_length=7,
                polyorder=1,
            )

        v["binned"] = v["binned"] / v["binned"].max(axis=1, keepdims=True)

    v["counts"] = np.array(counts).astype(np.int32)

    return v

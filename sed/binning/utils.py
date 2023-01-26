"""This file contains helper functions for the sed.binning module

"""
from typing import cast
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np


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
) -> Tuple[
    Union[List[int], List[np.ndarray]],
    Sequence[str],
    Sequence[Tuple[float, float]],
]:
    """Convert the flexible input for defining bins into a
    simple "axes" "bins" "ranges" tuple.

    This allows to mimic the input used in numpy.histogramdd flexibility into the
    binning functions defined here.

    Args:
        bins: Definition of the bins. Can  be any of the following cases:
            - an integer describing the number of bins in on all dimensions
            - a tuple of 3 numbers describing start, end and step of the binning range
            - a np.arrays defining the binning edges
            - a list (NOT a tuple) of any of the above (int, tuple or np.ndarray)
            - a dictionary made of the axes as keys and any of the above as values.
            This takes priority over the axes and range arguments. Defaults to 100
        axes: The names of the axes (columns) on which to calculate the histogram.
            The order will be the order of the dimensions in the resulting array.
            Defaults to None
        ranges: list of tuples containing the start and end point of the binning range.
            Defaults to None

    Returns:
        tuple containing axes, bins and ranges.
    """
    if isinstance(axes, str):
        axes = [axes]
    # if bins is a dictionary: unravel to axes and bins

    if isinstance(bins, dict):
        axes = []
        bins_ = []
        for k, v in bins.items():
            axes.append(k)
            bins_.append(v)
        bins = bins_
    elif isinstance(bins, (int, np.ndarray)):
        bins = [bins] * len(axes)
    elif isinstance(bins, tuple):
        if len(bins) == 3:
            bins = [bins]
        else:
            raise ValueError(
                "Bins defined as tuples should only be used to define start ",
                "stop and step of the bins. i.e. should always have lenght 3.",
            )
    if not isinstance(bins, list):
        raise TypeError(f"Cannot interpret bins of type {type(bins)}")
    if axes is None:
        raise AttributeError("Must define on which axes to bin")
    if not all(isinstance(x, type(bins[0])) for x in bins):
        raise TypeError('All elements in "bins" must be of the same type')

    if isinstance(bins[0], tuple):
        bins = cast(List[tuple], bins)
        assert len(bins[0]) == 3
        ranges = []
        bins_ = []
        for tpl in bins:
            assert isinstance(tpl, tuple)
            ranges.append((tpl[0], tpl[1]))
            bins_.append(int((tpl[1] - tpl[0])/tpl[2]))
        bins = bins_
    elif not isinstance(bins[0], (int, np.ndarray)):
        raise TypeError(f"Could not interpret bins of type {type(bins[0])}")

    if ranges is not None:
        if (len(axes) == len(bins) == 1) and isinstance(
            ranges[0],
            (int, float),
        ):
            ranges = (cast(Tuple[float, float], ranges),)
        elif not len(axes) == len(bins) == len(ranges):
            raise AttributeError(
                "axes and range and bins must have the same number of elements",
            )
    elif isinstance(bins[0], int):
        raise AttributeError(
            "Must provide a range if bins is an integer or list of integers",
        )
    elif len(axes) != len(bins):
        raise AttributeError(
            "axes and bins must have the same number of elements",
        )

    # TODO: mypy still thinks List[tuple] is a possible type for bins, nut sure why.
    bins = cast(Union[List[int], List[np.ndarray]], bins)

    return bins, axes, ranges


def bin_edges_to_bin_centers(bin_edges: np.ndarray) -> np.ndarray:
    """Converts a list of bin edges into corresponding bin centers

    Args:
        bin_edges: 1d array of bin edges

    Returns:
        bin_centers: 1d array of bin centers
    """

    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    return bin_centers


def bin_centers_to_bin_edges(bin_centers: np.ndarray) -> np.ndarray:
    """Converts a list of bin centers into corresponding bin edges

    Args:
        bin_centers: 1d array of bin centers

    Returns:
        bin_edges: 1d array of bin edges
    """
    bin_edges = (bin_centers[1:] + bin_centers[:-1]) / 2

    bin_edges = np.insert(
        bin_edges,
        0,
        bin_centers[0] - (bin_centers[1] - bin_centers[0]) / 2,
    )
    bin_edges = np.append(
        bin_edges,
        bin_centers[len(bin_centers) - 1]
        + (
            bin_centers[len(bin_centers) - 1]
            - bin_centers[len(bin_centers) - 2]
        )
        / 2,
    )

    return bin_edges

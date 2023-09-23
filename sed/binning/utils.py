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


def simplify_binning_arguments(
    bins: Union[
        int,
        dict,
        Sequence[int],
        Sequence[np.ndarray],
        Sequence[tuple],
    ],
    axes: Sequence[str] = None,
    ranges: Sequence[Tuple[float, float]] = None,
) -> Tuple[Union[List[int], List[np.ndarray]], List[str], List[Tuple[float, float]]]:
    """Convert the flexible input for defining bins into a
    simple "axes" "bins" "ranges" tuple.

    This allows to mimic the input used in numpy.histogramdd flexibility into the
    binning functions defined here.

    Args:
        bins (int, dict, Sequence[int], Sequence[np.ndarray], Sequence[tuple]):
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
        axes (Sequence[str], optional): Sequence containing the names of
            the axes (columns) on which to calculate the histogram. The order will be
            the order of the dimensions in the resulting array. Only not required if
            bins are provided as dictionary containing the axis names.
            Defaults to None.
        ranges (Sequence[Tuple[float, float]], optional): Sequence of tuples containing
            the start and end point of the binning range. Required if bins given as
            int or Sequence[int]. Defaults to None.

    Raises:
        ValueError: Wrong shape of bins,
        TypeError: Wrong type of bins
        AttributeError: Axes not defined
        AttributeError: Shape mismatch

    Returns:
        Tuple[Union[List[int], List[np.ndarray]], List[Tuple[float, float]]]: Tuple
        containing lists of bin centers, axes, and ranges.
    """
    # if bins is a dictionary: unravel to axes and bins
    if isinstance(bins, dict):
        axes = []
        bins_ = []
        for k, v in bins.items():
            axes.append(k)
            bins_.append(v)
        bins = bins_

    # if bins provided as single int, apply to all dimensions
    if isinstance(bins, (int, np.int64)):
        bins = [bins] * len(axes)

    # Check that we have a sequence of bins now
    if not isinstance(bins, Sequence):
        raise TypeError(f"Cannot interpret bins of type {type(bins)}")

    # check that we have axes
    if axes is None:
        raise AttributeError("Must define on which axes to bin")

    # check that axes is a sequence
    if not isinstance(axes, Sequence):
        raise TypeError(f"Cannot interpret axes of type {type(axes)}")

    # check that all elements of axes are str
    if not all(isinstance(axis, str) for axis in axes):
        raise TypeError("Axes has to contain only strings!")

    # we got tuples as bins, expand to bins and ranges
    if all(isinstance(x, tuple) for x in bins):
        bins = cast(Sequence[tuple], bins)
        assert len(bins[0]) == 3, "Tuples as bins need to have format (start, end, num_bins)."
        ranges = []
        bins_ = []
        for tpl in bins:
            assert isinstance(tpl, tuple)
            ranges.append((tpl[0], tpl[1]))
            bins_.append(tpl[2])
        bins = bins_

    # if bins are provided as int, check that ranges are present
    if all(isinstance(x, (int, np.int64)) for x in bins):
        bins = cast(List[int], list(bins))
        if ranges is None:
            raise AttributeError(
                "Must provide a range if bins is an integer or list of integers",
            )
        if not isinstance(ranges, Sequence):
            raise AttributeError(
                f"Ranges must be a sequence, not {type(ranges)}.",
            )

    # otherwise, all bins should by np.ndarrays here
    elif all(isinstance(x, np.ndarray) for x in bins):
        bins = cast(List[np.ndarray], list(bins))
    else:
        raise TypeError(f"Could not interpret bins of type {type(bins)}")

    # check that number of bins and number of axes is the same.
    if len(axes) != len(bins):
        raise AttributeError(
            "axes and bins must have the same number of elements",
        )

    return bins, list(axes), list(ranges) if ranges else None


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
        + (bin_centers[len(bin_centers) - 1] - bin_centers[len(bin_centers) - 2]) / 2,
    )

    return bin_edges

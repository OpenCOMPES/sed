import numpy as np


def get_linear_bin_edges(array: np.ndarray) -> np.ndarray:
    """returns the bin edges of the given array

    Args:
        array: the array of N center values from which to evaluate the bin range.
         Must be linear.

    Returns:
        edges: array of edges, with shape N+1
    """
    step = array[1] - array[0]
    last_step = array[-2] - array[-3]
    assert np.allclose(last_step, step), "not a linear array"
    return np.linspace(
        array[0] - step / 2,
        array[-1] + step / 2,
        array.size + 1,
    )

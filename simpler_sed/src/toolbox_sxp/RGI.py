# https://gist.github.com/Edenhofer/f248f0de5a1dce54a246375d53345bc5
import numpy as np
from jax import numpy as jnp


def _ndim_coords_from_arrays(points, ndim=None):
    """
    Convert a tuple of coordinate arrays to a (..., ndim)-shaped array.
    """
    if isinstance(points, tuple) and len(points) == 1:
        # handle argument tuple
        points = points[0]
    if isinstance(points, tuple):
        points = np.asanyarray(points).swapaxes(0, 1)
    else:
        points = np.asanyarray(points)
        if points.ndim == 1:
            if ndim is None:
                points = points.reshape(-1, 1)
            else:
                points = points.reshape(-1, ndim)
    return points


class _RegularGridInterpolator:
    # Based on SciPy's implementation which in turn is originally based on an
    # implementation by Johannes Buchner

    def __init__(
        self,
        points,
        values,
        method="linear",
        bounds_error=False,
        fill_value=jnp.nan,
    ):
        if method not in ("linear", "nearest"):
            raise ValueError(f"method {method!r} is not defined")
        self.method = method
        self.bounds_error = bounds_error

        if len(points) > values.ndim:
            ve = f"there are {len(points)} point arrays, but values has {values.ndim} dimensions"
            raise ValueError(ve)

        if hasattr(values, "dtype") and hasattr(values, "astype"):
            if not np.issubdtype(values.dtype, np.inexact):
                values = values.astype(float)

        self.fill_value = fill_value
        if fill_value is not None:
            fill_value_dtype = np.asarray(fill_value).dtype
            if hasattr(values, "dtype") and not np.can_cast(
                fill_value_dtype,
                values.dtype,
                casting="same_kind",
            ):
                raise ValueError(
                    "fill_value must be either 'None' or "
                    "of a type compatible with values",
                )

        for i, p in enumerate(points):
            if not np.all(np.diff(p) > 0.0):
                ve = f"the points in dimension {i} must be strictly ascending"
                raise ValueError(ve)
            if not np.asarray(p).ndim == 1:
                ve = f"the points in dimension {i} must be 1-dimensional"
                raise ValueError(ve)
            if not values.shape[i] == len(p):
                ve = f"there are {len(p)} points and {values.shape[i]} values in dimension {i}"
                raise ValueError(ve)
        if isinstance(points, jnp.ndarray):
            self.grid = points  # Do not unnecessarily copy arrays
        else:
            self.grid = tuple(jnp.asarray(p) for p in points)
        self.values = jnp.asarray(values)

    def __call__(self, xi, method=None):
        method = self.method if method is None else method
        if method not in ("linear", "nearest"):
            raise ValueError(f"method {method!r} is not defined")

        ndim = len(self.grid)
        #
        xi = _ndim_coords_from_arrays(xi, ndim=ndim)
        # SciPy performs some conversions here; skip those
        if xi.shape[-1] != len(self.grid):
            raise ValueError(
                "the requested sample points xi have dimension"
                f" {xi.shape[1]}, but this RegularGridInterpolator has"
                f" dimension {ndim}",
            )

        xi_shape = xi.shape
        xi = xi.reshape(-1, xi_shape[-1])

        if self.bounds_error:
            for i, p in enumerate(xi.T):
                p = xi[..., i]
                if not np.logical_and(
                    np.all(self.grid[i][0] <= p),
                    np.all(p <= self.grid[i][-1]),
                ):
                    ve = f"one of the requested xi is out of bounds in dimension {i}"
                    raise ValueError(ve)

        indices, norm_distances, out_of_bounds = self._find_indices(xi.T)
        if method == "linear":
            result = self._evaluate_linear(indices, norm_distances)
        elif method == "nearest":
            result = self._evaluate_nearest(indices, norm_distances)
        else:
            raise AssertionError("method must be bound")
        if not self.bounds_error and self.fill_value is not None:
            result = jnp.where(
                out_of_bounds.reshape(
                    result.shape[:1] + (1,) * (result.ndim - 1),
                ),
                self.fill_value,
                result,
            )

        return result.reshape(xi_shape[:-1] + self.values.shape[ndim:])

    def _evaluate_linear(self, indices, norm_distances):
        from itertools import product

        # slice for broadcasting over trailing dimensions in self.values
        vslice = (slice(None),) + (None,) * (self.values.ndim - len(indices))

        # find relevant values
        # each i and i+1 represents a edge
        edges = product(*[[i, i + 1] for i in indices])
        values = jnp.array(0.0)
        for edge_indices in edges:
            weight = jnp.array(1.0)
            for ei, i, yi in zip(edge_indices, indices, norm_distances):
                weight *= jnp.where(ei == i, 1 - yi, yi)
            values += self.values[edge_indices] * weight[vslice]
        return values

    def _evaluate_nearest(self, indices, norm_distances):
        idx_res = [
            jnp.where(yi <= 0.5, i, i + 1)
            for i, yi in zip(indices, norm_distances)
        ]
        return self.values[tuple(idx_res)]

    def _find_indices(self, xi):
        # find relevant edges between which xi are situated
        indices = []
        # compute distance to lower edge in unity units
        norm_distances = []
        # check for out of bounds xi
        out_of_bounds = jnp.zeros((xi.shape[1],), dtype=bool)
        # iterate through dimensions
        for x, grid in zip(xi, self.grid):
            i = jnp.searchsorted(grid, x) - 1
            i = jnp.where(i < 0, 0, i)
            i = jnp.where(i > grid.size - 2, grid.size - 2, i)
            indices.append(i)
            norm_distances.append((x - grid[i]) / (grid[i + 1] - grid[i]))
            if not self.bounds_error:
                out_of_bounds += x < grid[0]
                out_of_bounds += x > grid[-1]
        return indices, norm_distances, out_of_bounds

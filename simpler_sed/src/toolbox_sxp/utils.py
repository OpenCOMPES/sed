"""
Helpers functions for the toolbox_sxp package.

Inspired by the SED library:
<https://github.com/OpenCOMPES/sed>

Author: David Doblas Jiménez <david.doblas-jimenez@xfel.eu>
Copyright (c) 2023, European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.

You should have received a copy of the 3-Clause BSD License along with this
program. If not, see <https://opensource.org/licenses/BSD-3-Clause>
"""
import h5py
import numba
import numpy as np
import polars as pl
from cv2 import getRotationMatrix2D
from cv2 import transform
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from scipy.spatial import KDTree
from skimage import feature

from .RGI import _RegularGridInterpolator


def annotate_features(arr, threshold, order="cw", shifted=False):
    """Find blobs in the given array

    Blobs are found using the Laplacian of Gaussian (LoG) method.
    For each blob found, the method returns its coordinates and the standard
    deviation of the Gaussian kernel that detected the blob.

    Args:
        arr: numpy.ndarray
            2D array where blobs can be identified
        threshold: float
            Absolute lower bound for space maxima. Local maxima smaller than
            `threshold` are ignored. Reduce this to detect blobs with lower
            intensities
        order: str
            Reordering of the vertices, if any, in a clock-wise ("cw") or a
            counter clock-wise ("ccw") fashion
        shifted: bool
            False (default) assumes the array has not been shifted from its
            original location

    Returns:
        dict(str: numpy.ndarray)
            Dictionary with centre and a set vertices as keys. Each value
            represents the coordinates (y, x) and sigma of the blob
    """
    # rows, cols = arr.shape
    # xc, yc = (rows - 1) / 2, (cols - 1) / 2
    cols, rows = arr.shape
    xc, yc = (cols - 1) / 2.0, (rows - 1) / 2.0

    features = {}
    all_blobs = feature.blob_log(arr, threshold=threshold)

    central_blob = np.mean(all_blobs, axis=0)
    _, idx = KDTree(all_blobs).query(central_blob)
    # translate to ext
    if shifted:
        all_blobs[idx][0] = all_blobs[idx][0] - yc
        all_blobs[idx][1] = all_blobs[idx][1] - xc
    features["centre"] = all_blobs[idx]

    side_blobs = np.delete(all_blobs, idx, axis=0)
    side_blobs_mean = np.mean(side_blobs, axis=0)
    shifted_blobs = side_blobs - side_blobs_mean

    angles = np.arctan2(shifted_blobs[:, 1], shifted_blobs[:, 0])
    if order == "cw":
        order = np.argsort(angles)
    elif order == "ccw":
        order = np.argsort(angles)[::-1]

    if shifted:
        side_blobs[:, 0] -= yc
        side_blobs[:, 1] -= xc
    features["vertices"] = side_blobs[order]

    return features


def rotational_array_generation(
    features,
    symmetry_order=6,
    scale=1.0,
    export_centre=False,
):
    """Generate an ideal regular polygon from a given set of vertices

    Given a set of blobs and a symmetry order, it generates an ideal polygon
    for that symmetry by taking the centre and the first of the blobs as
    starting points.

    Args:
        features: dict (str:numpy.ndarray)
            Dictionary with centre and a set vertices as keys. Each value
            represents the coordinates (y, x) and sigma of the blob
        symmetry_order: int
            Symmetry of the polygon to generate. Default to 6
        scale: int or float
            Scaling factor with respect to the original positions of the blobs
        export_centre: bool
            False (default) do not to calculate the centre for the ideal
            symmetric polygon

    Returns:
        dict(str: numpy.ndarray)
            Dictionary with centre and a set vertices as keys. Each value
            represents the coordinates (y, x) of the blobs in the ideal polygon
            of symmetry `symmetry_order`

    """

    if symmetry_order not in [2, 3, 4, 6]:
        raise ValueError(
            "Symmetry order must be a proper rotation operation: e.g., 2, 3, 4 or 6",
        )

    targets = {}
    center = features["centre"][:-1]
    sym_positions = np.linspace(0, 360, symmetry_order, endpoint=False)
    scale_factor = np.zeros(sym_positions.size) + scale
    rot_array = np.zeros((sym_positions.size, 2))

    for idx, point in enumerate(sym_positions):
        rot_matrix = getRotationMatrix2D(center, point, scale_factor[idx])
        rot_array[idx] = transform(
            # use first point of the vertices as starting point
            features["vertices"][0, :-1].reshape(1, 1, -1),
            rot_matrix,
        ).squeeze()

    if export_centre:
        centre = np.array(
            (
                sum(i for i in rot_array[:, 0]) / symmetry_order,
                sum(i for i in rot_array[:, 1]) / symmetry_order,
            ),
        )
        targets["centre"] = centre

    targets["vertices"] = rot_array

    return targets


@numba.jit(parallel=True, nogil=True, nopython=True)
def jitter(size, distribution=None):
    """Generates an array of randomly distributed numbers

    The distribution can be `uniform`, where numbers are from (-1, 1) or
    `random`, with numbers from (-0.5, 0.5).

    Args:
        size: int or tuple
            Size of the array to be generated
        distribution: str
            `uniform` (default) distribution is used. `random` distribution is
            also available
    """
    if distribution == "random":
        return np.random.uniform(a=-0.5, b=0.5, size=size)

    return np.random.uniform(a=-1, b=1, size=size)


def generate_inverse_dfield(
    r_dfield,
    c_dfield,
    bin_ranges,
    detector_ranges,
):
    """Generate the inverse of the deformation field

    N-Dimensional linear interpolation of the deformation field coordinates
    made by Delaunay triangulation.

    Args:
        r_dfield: numpy.ndarray
            2D array of the row (x) coordinates of a deformation field
        c_dfield: numpy.ndarray
            2D array of the column (y) coordinates of a deformation field
        bin_ranges: ((int, int), (int, int))
            Ranges for bins
        detector ranges: ((int, int), (int, int))
            Ranges for detector

    Returns:
        numpy.ndarray
            Inverse deformation field interpolated to `detector_ranges` points
            for both x and y coordinates
    """
    r_mesh, c_mesh = np.meshgrid(
        np.linspace(
            detector_ranges[0][0],
            c_dfield.shape[0],
            detector_ranges[0][1],
            endpoint=False,
        ),
        np.linspace(
            detector_ranges[1][0],
            c_dfield.shape[1],
            detector_ranges[1][1],
            endpoint=False,
        ),
        sparse=False,
        indexing="ij",
    )

    bin_step = (
        np.asarray(bin_ranges)[0:2][:, 1] - np.asarray(bin_ranges)[0:2][:, 0]
    ) / c_dfield.shape

    rc_positions = np.zeros(
        (c_dfield.shape[0] * c_dfield.shape[1], c_dfield.ndim),
        dtype=np.float32,
    )

    rc_positions[:, 0] = r_dfield.ravel() + bin_ranges[0][0] / bin_step[0]
    rc_positions[:, 1] = c_dfield.ravel() + bin_ranges[0][0] / bin_step[1]

    r_dest = (
        bin_step[0]
        * np.repeat(np.arange(c_dfield.shape[0]), c_dfield.shape[1])
        + bin_ranges[0][0]
    )
    c_dest = (
        bin_step[1] * np.tile(np.arange(c_dfield.shape[1]), c_dfield.shape[0])
        + bin_ranges[1][0]
    )

    # Build the triangulator upfront
    tri = Delaunay(rc_positions)

    linterp = LinearNDInterpolator(tri, r_dest)
    inv_r_dfield = linterp((r_mesh, c_mesh))

    linterp = LinearNDInterpolator(tri, c_dest)
    inv_c_dfield = linterp((r_mesh, c_mesh))

    return np.stack((inv_r_dfield, inv_c_dfield), axis=0)


def apply_dfield(
    df,
    dfield,
    x_col,
    y_col,
    detector_ranges,
):
    """Apply a deformation field

    A deformation field, previously calculated, is applied to the `x_col` and
    `y_col` of a dataframe `df`

    Args:
        df: polars.DataFrame
            Dataframe with, at least, `x_col` and `y_col`
        dfield: numpy.ndarray
            (Inverse) deformation field
        x_col: str
            DataFrame column to be used in the calculations
        y_col: str
            DataFrame column to be used in the calculations
        detector ranges: ((int, int), (int, int))
            Ranges for detector

    Returns:
        polars.DataFrame
            Dataframe with `Xm` and `Ym` columns resulting from the application
            of a deformation field to `x_col` and `y_col`, respectively
    """
    x, y = df[[x_col, y_col]]

    r_axis = np.linspace(
        detector_ranges[0][0],
        dfield[0].shape[0],
        detector_ranges[0][1],
        endpoint=False,
    )
    c_axis = np.linspace(
        detector_ranges[1][0],
        dfield[0].shape[1],
        detector_ranges[1][1],
        endpoint=False,
    )

    interp_x = _RegularGridInterpolator(
        (r_axis, c_axis),
        dfield[0],
        bounds_error=False,
    )
    xm = interp_x((x, y))

    interp_y = _RegularGridInterpolator(
        (r_axis, c_axis),
        dfield[1],
        bounds_error=False,
    )
    ym = interp_y((x, y))

    deformation_df = pl.DataFrame(
        {
            "Xm": np.array(xm).T,
            "Ym": np.array(ym).T,
        },
    )

    return deformation_df


def concat_dfs(dfs):
    """Concatenate dataframes horizontally

    Args:
        dfs: tuple or list
            Iterable of dataframes

    Returns:
        polars.DataFrame
    """
    return pl.concat(list(dfs), how="horizontal", rechunk=True)


def _find_nearest(val, arr):
    """Find the closest index value to a given one in an array

    Args:
        val: int or float
            Value of interest
        arr: numpy.ndarray
            Array to look for the nearest value

    Returns:
        int
            Index of the value nearest to the given one
    """
    idx = np.searchsorted(arr, val)
    if idx == 0:
        return idx

    if idx == arr.size:
        return idx - 1

    return idx - 1 if (val - arr[idx - 1]) < (arr[idx] - val) else idx


def _range_convert(arr, rng, path):
    """
    Convert value range using a pairwise path correspondence (e.g. obtained
    from dynamic time warping techniques (DTW)).

    Args:
        x_arr: numpy.array
            1D array
        x_rng: (int, int)
            Range containing a feature
        path: numpy.ndarray
            2D array containing a pair (x, y) of a DTW path

    Returns:
        (float, float)
            Tuple with a transformed range according to the path correspondence
    """
    return tuple(
        arr[path[_find_nearest(_find_nearest(value, arr), path[:, 0]), 1]]
        for value in rng
    )


def read_binned_h5(filename, combined=True):
    """Read files after being previously binned

    Returns a dictionary with two keys: axes and binned.

    Args:
        filename: str
            A path to and HDF5 file
        combined: bool
            True (default) combine the binned array into a numpy ndarray
    """
    data = {}
    with h5py.File(filename, "r") as h5:
        data["axes"] = {k: v[:] for k, v in h5["axes"].items()}
        if combined:
            data["binned"] = np.asarray(
                [h5["binned"][k] for k in h5["binned"].keys()],
            )
        else:
            data["binned"] = {k: v[:] for k, v in h5["binned"].items()}

    return data

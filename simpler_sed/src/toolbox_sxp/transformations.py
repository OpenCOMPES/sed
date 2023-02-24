"""
Image transformation functions for the toolbox_sxp package.

Inspired by the SED library:
<https://github.com/OpenCOMPES/sed>

Author: David Doblas Jiménez <david.doblas-jimenez@xfel.eu>
Copyright (c) 2023, European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.

You should have received a copy of the 3-Clause BSD License along with this
program. If not, see <https://opensource.org/licenses/BSD-3-Clause>
"""
import cv2
import numpy as np


def _get_extent(arr):
    """Get the extent of an array

    Args:
        arr: numpy.ndarray

    Returns:
        int, int, (float, float, float, float)
        Tuple the number of columns, rows and (-x, x, -y, y) coordinates of the
        bounding box of an image
    """
    if not isinstance(arr, np.ndarray) and arr.ndim != 2:
        raise ValueError(
            f"Array must be 2 dimensional. It has {arr.ndim} dimensions",
        )

    cols, rows = arr.shape
    xc, yc = (cols - 1) / 2.0, (rows - 1) / 2.0
    return cols, rows, (-xc, xc, -yc, yc)


def translate(arr, shift=(0.0, 0.0)):
    """Translate an array to a given point

    Given a `shift` point, the returned array is translated and centered
    on that point.

    Args:
        arr: numpy.ndarray
            2D array representing an image
        shift: (float, float)
            Coordinates (y, x) where the array should be centered

    Returns:
        (numpy.ndarray, (float, float, float, float))
            2D array translated to `shift` coordinates and an extent
            (-xc, xc, -yc, yc) object to be used for plotting as a bounding box
    """
    cols, rows, extent = _get_extent(arr)

    y_shift, x_shift = shift
    M = np.float32(
        [[1.0, 0.0, cols / 2.0 - x_shift], [0.0, 1.0, rows / 2.0 - y_shift]],
    )
    dst = cv2.warpAffine(arr, M, (cols, rows))

    return dst, extent


def rotate(arr, angle=0.0, rotation_center=(0, 0), scaling=1.0):
    """Rotate an array a given angle around a rotation center

    Given an `angle`, the returned array is rotated, and scaled, around its
    `rotation_center` point.

    Usually, this operation is applied after a `translate` operation, so the
    image has been previously centered at (0., 0.).

    Args:
        arr: numpy.ndarray
            2D array representing an image
        angle: float
            Rotation angle in degrees
        rotation_center: (float, float)
            Coordinates (y, x) from where the image will be rotated
        scale: float
            Factor applied to the image to zoom in (scale > 1.) or zoom out
            (scale < 1.)

    Returns:
        (numpy.ndarray, (float, float, float, float))
            2D array rotated `angle` degrees around `rotation_center` and an
            extent (-xc, xc, -yc, yc) object to be used for plotting as a
            bounding box
    """
    cols, rows, extent = _get_extent(arr)

    x_rot, y_rot = rotation_center
    M = cv2.getRotationMatrix2D(
        center=(x_rot, y_rot),
        angle=angle,
        scale=scaling,
    )
    dst = cv2.warpAffine(arr, M, (cols, rows))

    return dst, extent


def scale(arr, x_scale=None, y_scale=None):
    """Scales an array along x and/or y coordinates

    Usually, this operation is applied after a `translate` operation, so the
    image has been previously centered at (0., 0.).

    Args:
        arr: numpy.ndarray
            2D array representing an image
        x_scale: float
            Factor applied to the image along the x coordinate to zoom in
            (scale > 1.) or zoom out (scale < 1.)
        y_scale: float
            Factor applied to the image along the y coordinate to zoom in
            (scale > 1.) or zoom out (scale < 1.)

    Returns:
        numpy.ndarray
            2D array rescaled
    """

    dst = cv2.resize(
        arr,
        None,
        fx=x_scale,
        fy=y_scale,
        interpolation=cv2.INTER_CUBIC,
    )

    return dst


def translate_and_rotate(
    arr,
    shift=(0.0, 0.0),
    angle=0.0,
    rotation_center=(0.0, 0.0),
    scaling=1.0,
):
    """Translate an array to a given point and rotate that array a given angle
    around a rotation center

    Operation that first translate and then rotate, and scale, an array around
    its `rotation_center` point. The resulting array will be centered on that
    point.

    Args:
        arr: numpy.ndarray
            2D array representing an image
        shift: (float, float)
            Coordinates (y, x) where the array should be centered
        angle: float
            Rotation angle in degrees
        rotation_center: (float, float)
            Coordinates (y, x) from where the image will be rotated
        scale: float
            Factor applied to the image to zoom in (scale > 1.) or zoom out
            (scale < 1.)

    Returns:
        (numpy.ndarray, (float, float, float, float))
            2D array translated to `shift` and rotated `angle` degrees around
            `rotation_center` and an extent (-xc, xc, -yc, yc) object to be
            used for plotting as a bounding box
    """
    cols, rows, extent = _get_extent(arr)

    y_shift, x_shift = shift

    shift_matrix = np.float32(
        [
            [1.0, 0.0, cols / 2.0 - x_shift],
            [0.0, 1.0, rows / 2.0 - y_shift],
            [0.0, 0.0, 1.0],
        ],
    )

    y_rot, x_rot = rotation_center

    rotation_matrix = cv2.getRotationMatrix2D(
        center=(x_rot, y_rot),
        angle=angle,
        scale=scaling,
    )
    rotation_matrix = np.vstack([rotation_matrix, [0.0, 0.0, 1.0]])

    M = np.matmul(shift_matrix, rotation_matrix)

    # warpPerspective is used here, because the matrix is now 3x3 not 3x2
    dst = cv2.warpPerspective(arr, M, (cols, rows), flags=cv2.INTER_LANCZOS4)

    return dst, extent


def generate_splinewarp(arr, features, targets, get_dfield=True):
    """Generate spline warp correction from a set of features

    Args:
        arr: numpy.ndarray
            2D array representing an image
        features: dict(str: numpy.ndarray)
            Dictionary with centre and a set vertices as keys. Each value
            represents the coordinates (y, x) and sigma of the blob found in
            the original array
        targets: dict(str: numpy.ndarray)
            Dictionary with centre and a set vertices as keys. Each value
            represents the coordinates (y, x) of an ideal blob
        get_dfield: bool
            True (default) will return the deformation field (x, y). Each of
            the coordinates has been reshaped to `arr.size`

    Returns:
        (numpy.ndarray, Optional((numpy.ndarray, numpy.ndarray))
            2D array warped according the pre-estimated transformation
            parameters, and its corresponding deformation field (optional)
    """
    tps = cv2.createThinPlateSplineShapeTransformer()

    _sources = np.append(
        features["vertices"][:, :2],
        features["centre"][:2],
    ).reshape(1, -1, 2)

    _targets = np.append(targets["vertices"], targets["centre"]).reshape(
        1,
        -1,
        2,
    )

    # number of features plus the center
    nfeatures = sum(len(val.flatten()) // 3 for val in features.values())
    _matches = [cv2.DMatch(f, f, 0) for f in range(nfeatures)]

    tps.estimateTransformation(_sources, _targets, _matches)
    warped = tps.warpImage(
        arr,
        np.zeros_like(arr),
        cv2.INTER_NEAREST,
        cv2.BORDER_REPLICATE,
    )

    if get_dfield:
        width, height = arr.shape
        x, y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
            copy=False,
        )
        def_stack = np.dstack((x.ravel(), y.ravel()))
        _transformation_cost, def_field = tps.applyTransformation(def_stack)

        xy = def_field.reshape(width, height, 2)
        x_def, y_def = xy[..., 0], xy[..., 1]

        return warped, (x_def, y_def)

    return warped

# Note: some of the functions presented here were
# inspired by https://github.com/mpes-kit/mpes
from typing import Tuple
from typing import Union

import dask.dataframe
import numpy as np
import pandas as pd
from scipy.interpolate import griddata


def apply_distortion_correction(
    df: Union[pd.DataFrame, dask.dataframe.DataFrame],
    X: str = "X",
    Y: str = "Y",
    newX: str = "Xm",
    newY: str = "Ym",
    type: str = "mattrans",
    **kwds: dict,
) -> Union[pd.DataFrame, dask.dataframe.DataFrame]:
    """Calculate and replace the X and Y values with their distortion-corrected version.
    This method can be reused.

    :Parameters:
        df : dataframe
            Dataframe to apply the distotion correction to.
        X, Y: | 'X', 'Y'
            Labels of the columns before momentum distortion correction.
        newX, newY: | 'Xm', 'Ym'
            Labels of the columns after momentum distortion correction.

    :Return:
        dataframe with added columns
    """

    if type == "mattrans":  # Apply matrix transform
        if "warping" in kwds:
            warping = kwds.pop("warping")
            df[newX], df[newY] = perspective_transform(
                df[X],
                df[Y],
                warping,
                **kwds,
            )
            return df
        else:
            raise NotImplementedError
    elif type == "tps" or type == "tps_matrix":
        if "dfield" in kwds:
            dfield = kwds.pop("dfield")
            out_df = df.map_partitions(
                apply_dfield,
                dfield,
                X=X,
                Y=Y,
                newX=newX,
                newY=newY,
                **kwds,
            )
            return out_df
        elif "rdeform_field" in kwds and "cdeform_field" in kwds:
            rdeform_field = kwds.pop("rdeform_field")
            cdeform_field = kwds.pop("cdeform_field")
            print(
                "Calculating inverse Deformation Field, might take a moment...",
            )
            dfield = generate_dfield(rdeform_field, cdeform_field)
            out_df = df.map_partitions(
                apply_dfield,
                dfield,
                X=X,
                Y=Y,
                newX=newX,
                newY=newY,
                **kwds,
            )
            return out_df
        else:
            raise NotImplementedError


def append_k_axis(
    df: Union[pd.DataFrame, dask.dataframe.DataFrame],
    x0: float,
    y0: float,
    X: str = "X",
    Y: str = "Y",
    newX: str = "kx",
    newY: str = "ky",
    **kwds: dict,
) -> Union[pd.DataFrame, dask.dataframe.DataFrame]:
    """Calculate and append the k axis coordinates (kx, ky) to the events dataframe.
    This method can be reused.

    :Parameters:
        df : dataframe
            Dataframe to apply the distotion correction to.
        x0:
            center of the k-image in row pixel coordinates
        y0:
            center of the k-image in column pixel coordinates
        X, Y: | 'X', 'Y'
            Labels of the source columns
        newX, newY: | 'ky', 'ky'
            Labels of the destination columns
        **kwds:
            additional keywords for the momentum conversion

    :Return:
        dataframe with added columns
    """

    if "fr" in kwds and "fc" in kwds:
        df[newX], df[newY] = detector_coordiantes_2_k_koordinates(
            rdet=df[X],
            cdet=df[Y],
            r0=x0,
            c0=y0,
            **kwds,
        )
        return df

    else:
        raise NotImplementedError


def detector_coordiantes_2_k_koordinates(
    rdet: float,
    cdet: float,
    rstart: float,
    cstart: float,
    r0: float,
    c0: float,
    fr: float,
    fc: float,
    rstep: float,
    cstep: float,
) -> Tuple[float, float]:
    """
    Conversion from detector coordinates (rdet, cdet) to momentum coordinates (kr, kc).
    """

    rdet0 = rstart + rstep * r0
    cdet0 = cstart + cstep * c0
    kr = fr * ((rdet - rdet0) / rstep)
    kc = fc * ((cdet - cdet0) / cstep)

    return (kr, kc)


def apply_dfield(
    df: Union[pd.DataFrame, dask.dataframe.DataFrame],
    dfield: np.ndarray,
    X: str = "X",
    Y: str = "Y",
    newX: str = "Xm",
    newY: str = "Ym",
) -> Union[pd.DataFrame, dask.dataframe.DataFrame]:
    """
    Application of the inverse displacement-field to the dataframe coordinates

    :Parameters:
        df : dataframe
            Dataframe to apply the distotion correction to.
        dfield:
            The distortion correction field. 3D matrix,
            with column and row distortion fields stacked along the first dimension.
        X, Y: | 'X', 'Y'
            Labels of the source columns
        newX, newY: | 'Xm', 'Ym'
            Labels of the destination columns

    :Return:
        dataframe with added columns
    """

    x = df[X]
    y = df[Y]

    df[newX], df[newY] = (
        dfield[0, np.int16(x), np.int16(y)],
        dfield[1, np.int16(x), np.int16(y)],
    )
    return df


def generate_dfield(
    rdeform_field: np.ndarray,
    cdeform_field: np.ndarray,
) -> np.ndarray:
    """
    Generate inverse deformation field using inperpolation with griddata.
    Assuming the binning range of the input ``rdeform_field`` and ``cdeform_field``
    covers the whole detector.

    :Parameters:
        rdeform_field, cdeform_field: 2d array, 2d array
            Row-wise and column-wise deformation fields.

    :Return:
        dfield:
            inverse 3D deformation field stacked along the first dimension
    """
    # Interpolate to 2048x2048 grid of the detector coordinates
    grid_x, grid_y = np.mgrid[
        0 : cdeform_field.shape[0] : (cdeform_field.shape[0] / 2048),
        0 : cdeform_field.shape[1] : (cdeform_field.shape[1] / 2048),
    ]
    XY = []
    Z = []
    for i in np.arange(cdeform_field.shape[0]):
        for j in np.arange(cdeform_field.shape[1]):
            XY.append([rdeform_field[i, j], cdeform_field[i, j]])
            Z.append(2048 / cdeform_field.shape[0] * i)

    inv_rdeform_field = griddata(np.asarray(XY), Z, (grid_x, grid_y))

    XY = []
    Z = []
    for i in np.arange(cdeform_field.shape[0]):
        for j in np.arange(cdeform_field.shape[1]):
            XY.append([rdeform_field[i, j], cdeform_field[i, j]])
            Z.append(2048 / cdeform_field.shape[1] * j)

    inv_cdeform_field = griddata(np.asarray(XY), Z, (grid_x, grid_y))

    # TODO: what to do about the nans at the boundary? leave or fill with zeros?
    # inv_rdeform_field = np.nan_to_num(inv_rdeform_field)
    # inv_rdeform_field = np.nan_to_num(inv_cdeform_field)

    dfield = np.asarray([inv_rdeform_field, inv_cdeform_field])

    return dfield


def perspective_transform(
    x: float,
    y: float,
    M: np.ndarray,
) -> Tuple[float, float]:
    """Implementation of the perspective transform (homography) in 2D.

    :Parameters:
        x, y: numeric, numeric
            Pixel coordinates of the original point.
        M: 2d array
            Perspective transform matrix.

    :Return:
        xtrans, ytrans: numeric, numeric
            Pixel coordinates after projective/perspective transform.
    """

    denom = M[2, 0] * x + M[2, 1] * y + M[2, 2]
    xtrans = (M[0, 0] * x + M[0, 1] * y + M[0, 2]) / denom
    ytrans = (M[1, 0] * x + M[1, 1] * y + M[1, 2]) / denom

    return xtrans, ytrans

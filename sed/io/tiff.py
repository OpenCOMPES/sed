"""This module contains tiff file input/output functions for the sed.io module

"""
from pathlib import Path
from typing import Sequence
from typing import Union

import numpy as np
import tifffile
import xarray as xr


_IMAGEJ_DIMS_ORDER = "TZCYXS"
_IMAGEJ_DIMS_ALIAS = {
    "T": [
        "delayStage",
        "pumpProbeTime",
        "time",
        "delay",
        "T",
    ],
    "Z": [
        "dldTime",
        "t",
        "energy",
        "e",
        "E",
        "binding_energy",
        "energies",
        "binding_energies",
    ],
    "C": ["C"],
    "Y": ["dldPosY", "ky", "y", "ypos", "Y"],
    "X": ["dldPosX", "kx", "x", "xpos", "X"],
    "S": ["S"],
}


def to_tiff(
    data: Union[xr.DataArray, np.ndarray],
    faddr: Union[Path, str],
    alias_dict: dict = None,
):
    """Save an array as a .tiff stack compatible with ImageJ

    Args:
        data (Union[xr.DataArray, np.ndarray]): data to be saved. If a np.ndarray,
            the order is retained. If it is an xarray.DataArray, the order is inferred
            from axis_dict instead. ImageJ likes tiff files with axis order as
            TZCYXS. Therefore, best axis order in input should be: Time, Energy,
            posY, posX. The channels 'C' and 'S' are automatically added and can
            be ignored.
        faddr (Union[Path, str]): full path and name of file to save.
        alias_dict (dict, optional): name pairs for correct axis ordering. Keys should
            be any of T,Z,C,Y,X,S. The Corresponding value should be a dimension of the
            xarray or the dimension number if a numpy array. This is used to sort the
            data in the correct order for imagej standards. If None it tries to guess
            the order from the name of the axes or assumes T,Z,C,Y,X,S order for numpy
            arrays. Defaults to None.

    Raises:
        AttributeError: if more than one axis corresponds to a single dimension
        NotImplementedError: if data is not 2,3 or 4 dimensional
        TypeError: if data is not a np.ndarray or an xarray.DataArray
    """
    out: Union[np.ndarray, xr.DataArray] = None
    if isinstance(data, np.ndarray):
        # TODO: add sorting by dictionary keys
        dim_expansions = {2: [0, 1, 2, 5], 3: [0, 2, 5], 4: [2, 5]}
        dims = {
            2: ["x", "y"],
            3: ["x", "y", "energy"],
            4: ["x", "y", "energy", "delay"],
        }
        try:
            out = np.expand_dims(data, dim_expansions[data.ndim])
        except KeyError as exc:
            raise NotImplementedError(
                f"Only 2-3-4D arrays supported when data is a {type(data)}",
            ) from exc

        dims_order = dims[data.ndim]

    elif isinstance(data, xr.DataArray):
        dims_order = _fill_missing_dims(list(data.dims), alias_dict=alias_dict)
        out = data.expand_dims(
            {dim: 1 for dim in dims_order if dim not in data.dims},
        )
        out = out.transpose(*dims_order)
    else:
        raise TypeError(f"Cannot handle data of type {data.type}")

    faddr = Path(faddr).with_suffix(".tiff")

    tifffile.imwrite(faddr, out.astype(np.float32), imagej=True)

    print(f"Successfully saved {faddr}\n Axes order: {dims_order}")


def _sort_dims_for_imagej(dims: Sequence, alias_dict: dict = None) -> list:
    """Guess the order of the dimensions from the alias dictionary.

    Args:
        dims (Sequence): the list of dimensions to sort
        alias_dict (dict, optional): name pairs for correct axis ordering. Keys should
            be any of T,Z,C,Y,X,S. The Corresponding value should be a dimension of the
            xarray or the dimension number if a numpy array. This is used to sort the
            data in the correct order for imagej standards. If None it tries to guess
            the order from the name of the axes or assumes T,Z,C,Y,X,S order for numpy
            arrays. Defaults to None.

    Raises:
        ValueError: for duplicate entries for a single imagej dimension
        NameError: when a dimension cannot be found in the alias dictionary

    Returns:
        list: List of sorted dimension names.
    """
    order = _fill_missing_dims(dims=dims, alias_dict=alias_dict)
    return [d for d in order if d in dims]


def _fill_missing_dims(dims: Sequence, alias_dict: dict = None) -> list:
    """Fill in the missing dimensions from the alias dictionary.

    Args:
        dims (Sequence): the list of dimensions that are provided
        alias_dict (dict, optional): name pairs for correct axis ordering. Keys should
            be any of T,Z,C,Y,X,S. The Corresponding value should be a dimension of the
            xarray or the dimension number if a numpy array. This is used to sort the
            data in the correct order for imagej standards. If None it tries to guess
            the order from the name of the axes or assumes T,Z,C,Y,X,S order for numpy
            arrays. Defaults to None.

    Raises:
        ValueError: for duplicate entries for a single imagej dimension
        NameError: when a dimension cannot be found in the alias dictionary

    Returns:
        list: augmented list of TIFF dimensions.
    """
    order: list = []
    # overwrite the default values with the provided dict
    if alias_dict is None:
        alias_dict = {}
    else:
        for k, v in alias_dict.items():
            assert k in _IMAGEJ_DIMS_ORDER, f"keys must all be one of {_IMAGEJ_DIMS_ALIAS}"
            if not isinstance(v, (list, tuple)):
                alias_dict[k] = [v]

    alias_dict = {**_IMAGEJ_DIMS_ALIAS, **alias_dict}
    added_dims = 0
    for imgj_dim in _IMAGEJ_DIMS_ORDER:
        found_one = False
        for dim in dims:
            if dim in alias_dict[imgj_dim]:
                if found_one:
                    raise ValueError(
                        f"Duplicate entries for {imgj_dim}: {dim} and {order[-1]} ",
                    )
                order.append(dim)
                found_one = True
        if not found_one:
            order.append(imgj_dim)
            added_dims += 1
    if len(order) != len(dims) + added_dims:
        raise NameError(
            f"Could not interpret dimensions {[d for d in dims if d not in order]}",
        )
    return order


def load_tiff(
    faddr: Union[str, Path],
    coords: dict = None,
    dims: Sequence = None,
    attrs: dict = None,
) -> xr.DataArray:
    """Loads a tiff stack to an xarray.

    The .tiff format does not retain information on the axes, so these need to
    be manually added with the axes argument. Otherwise, this returns the data
    only as np.ndarray.

    Args:
        faddr (Union[str, Path]): Path to file to load.
        coords (dict, optional): The axes describing the data, following the tiff
            stack order. Defaults to None.
        dims (Sequence, optional): the order of the coordinates provided, considering
            the data is ordered as TZCYXS. If None (default) it infers the order from
            the order of the coords dictionary.
        attrs (dict, optional): dictionary to add as attributes to the
            xarray.DataArray. Defaults to None.

    Returns:
        xr.DataArray: an xarray representing the data loaded from the .tiff file
    """
    data = tifffile.imread(faddr)

    if coords is None:
        coords = {
            k.replace("_", ""): np.linspace(0, n, n)
            for k, n in zip(
                _IMAGEJ_DIMS_ORDER,
                data.shape,
            )
            if n > 1
        }

    data = data.squeeze()

    if dims is None:
        dims = list(coords.keys())

    assert data.ndim == len(dims), (
        f"Data dimension {data.ndim} must coincide number of coordinates "
        f"{len(coords)} and dimensions {len(dims)} provided,"
    )
    return xr.DataArray(data=data, coords=coords, dims=dims, attrs=attrs)

from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Sequence
from typing import Tuple
from typing import Union

import h5py
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


def recursive_write_metadata(h5group: h5py.Group, node: dict):
    """Recurses through a python dictionary and writes it into an hdf5 file.

    Args:
        h5group: hdf5 group element where to store the current dict node to.
        node: dictionary node to store

    Raises:
        Warning: warns if elements have been converted into strings for saving.
        ValueError: Rises when elements cannot be saved even as strings.
    """
    for key, item in node.items():
        if isinstance(
            item,
            (np.ndarray, np.int64, np.float64, str, bytes, int, float, list),
        ):
            try:
                h5group.create_dataset(key, data=item)
            except TypeError:
                h5group.create_dataset(key, data=str(item))
                print(f"Saved {key} as string.")
        elif isinstance(item, dict):
            print(key)
            group = h5group.create_group(key)
            recursive_write_metadata(group, item)
        else:
            try:
                h5group.create_dataset(key, data=str(item))
                print(f"Saved {key} as string.")
            except BaseException as exc:
                raise Exception(
                    f"Unknown error occured, cannot save {item} of type {type(item)}.",
                ) from exc


def recursive_parse_metadata(
    node: Union[h5py.Group, h5py.Dataset],
) -> dict:
    """Recurses through an hdf5 file, and parse it into a dictionary.

    Args:
        node: hdf5 group or dataset to parse into dictionary.

    Returns:
        dictionary: Dictionary of elements in the hdf5 path contained in node
    """
    if isinstance(node, h5py.Group):
        dictionary = {}
        for key, value in node.items():
            dictionary[key] = recursive_parse_metadata(value)

    else:
        entry = node[...]
        try:
            dictionary = entry.item()
            if isinstance(dictionary, (bytes, bytearray)):
                dictionary = dictionary.decode()
        except ValueError:
            dictionary = entry

    return dictionary


def to_h5(data: xr.DataArray, faddr: str, mode: str = "w"):
    """Save xarray formatted data to hdf5

    Args:
        data: input data
        faddr (str): complete file name (including path)
        mode (str): hdf5 read/write mode

    Raises:
        Warning: subfunction warns if elements have been converted into strings for
        saving.

    Returns:
    """
    with h5py.File(faddr, mode) as h5_file:

        print(f"saving data to {faddr}")

        # Saving data, make a single dataset
        dataset = h5_file.create_dataset("binned/BinnedData", data=data.data)
        try:
            dataset.attrs["units"] = data.attrs["units"]
            dataset.attrs["long_name"] = data.attrs["long_name"]
        except KeyError:
            pass

        # Saving axes
        axes_group = h5_file.create_group("axes")
        axes_number = 0
        for bin_name in data.dims:
            axis = axes_group.create_dataset(
                f"ax{axes_number}",
                data=data.coords[bin_name],
            )
            axis.attrs["name"] = bin_name
            try:
                axis.attrs["unit"] = data.coords[bin_name].attrs["unit"]
            except KeyError:
                pass
            axes_number += 1

        if "metadata" in data.attrs and isinstance(
            data.attrs["metadata"],
            dict,
        ):
            meta_group = h5_file.create_group("metadata")

            recursive_write_metadata(meta_group, data.attrs["metadata"])

    print("Saving complete!")


def load_h5(faddr: str, mode: str = "r") -> xr.DataArray:
    """Read xarray data from formatted hdf5 file

    Args:
        faddr: complete file name (including path)
        mode: hdf5 read/write mode

    Returns:
        xarray: output xarra data
    """
    with h5py.File(faddr, mode) as h5_file:
        # Reading data array
        try:
            data = h5_file["binned"]["BinnedData"]
        except KeyError as exc:
            raise Exception(
                "Wrong Data Format, the BinnedData were not found. "
                f"The error was{exc}.",
            ) from exc

        # Reading the axes
        bin_axes = []
        bin_names = []

        try:
            for axis in h5_file["axes"]:
                bin_axes.append(h5_file["axes"][axis])
                bin_names.append(h5_file["axes"][axis].attrs["name"])
        except KeyError as exc:
            raise Exception(
                f"Wrong Data Format, the axes were not found. The error was {exc}",
            ) from exc

        # load metadata
        metadata = None
        if "metadata" in h5_file:
            metadata = recursive_parse_metadata(h5_file["metadata"])

        coords = {}
        for name, vals in zip(bin_names, bin_axes):
            coords[name] = vals

        xarray = xr.DataArray(data, dims=bin_names, coords=coords)

        try:
            for axis in bin_axes:
                xarray[bin_names[axis]].attrs["unit"] = h5_file["axes"][
                    axis
                ].attrs["unit"]
            xarray.attrs["units"] = h5_file["binned"]["BinnedData"].attrs[
                "units"
            ]
            xarray.attrs["long_name"] = h5_file["binned"]["BinnedData"].attrs[
                "long_name"
            ]
        except (KeyError, TypeError):
            pass

        if metadata is not None:
            xarray.attrs["metadata"] = metadata

        return xarray


def to_tiff(
    data: Union[xr.DataArray, np.ndarray],
    faddr: Union[Path, str],
    alias_dict: dict = None,
) -> None:
    """Save an array as a .tiff stack compatible with ImageJ

    Args:
        data: data to be saved. If a np.ndarray, the order is retained. If it
            is an xarray.DataArray, the order is inferred from axis_dict instead.
            ImageJ likes tiff files with axis order as
            TZCYXS. Therefore, best axis order in input should be: Time, Energy,
            posY, posX. The channels 'C' and 'S' are automatically added and can
            be ignored.
        faddr: full path and name of file to save.
        alias_dict: name pairs for correct axis ordering. Keys should be any of
            T,Z,C,Y,X,S. The Corresponding value should be a dimension of the xarray or
            the dimension number if a numpy array. This is used to sort the data in the
            correct order for imagej standards. If None it tries to guess the order
            from the name of the axes or assumes T,Z,C,Y,X,S order for numpy arrays.
            Defaults to None

    Raise:
        AttributeError: if more than one axis corresponds to a single dimension
        NotImplementedError: if data is not 2,3 or 4 dimensional
        TypeError: if data is not a np.ndarray or an xarray.DataArray
    """

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
        except KeyError:
            raise NotImplementedError(
                f"Only 2-3-4D arrays supported when data is a {type(data)}",
            )

        dims_order = dims[data.ndim]

    elif isinstance(data, xr.DataArray):
        dims_order = _fill_missing_dims(data.dims, alias_dict=alias_dict)
        out = data.expand_dims(
            {dim: 1 for dim in dims_order if dim not in data.dims},
        )
        out = out.transpose(*dims_order)
    else:
        raise TypeError(f"Cannot handle data of type {data.type}")

    faddr = Path(faddr).with_suffix(".tiff")

    tifffile.imwrite(faddr, out.astype(np.float32), imagej=True)

    print(f"Successfully saved {faddr}\n Axes order: {dims_order}")


def _sort_dims_for_imagej(dims: list, alias_dict: dict = None) -> list:
    """Guess the order of the dimensions from the alias dictionary

    Args:
        dims: the list of dimensions to sort

    Raises:
        ValueError: for duplicate entries for a single imagej dimension
        NameError: when a dimension cannot be found in the alias dictionary

    Returns:
        _description_
    """
    order = _fill_missing_dims(dims=dims, alias_dict=alias_dict)
    return [d for d in order if d in dims]


def _fill_missing_dims(dims: list, alias_dict: dict = None) -> list:
    """Guess the order of the dimensions from the alias dictionary

    Args:
        dims: the list of dimensions to sort

    Raises:
        ValueError: for duplicate entries for a single imagej dimension
        NameError: when a dimension cannot be found in the alias dictionary

    Returns:
        _description_
    """
    order = []
    # overwrite the default values with the provided dict
    if alias_dict is None:
        alias_dict = {}
    else:
        for k, v in alias_dict.items():
            assert k in _IMAGEJ_DIMS_ORDER, (
                "keys must all be one of " f"{_IMAGEJ_DIMS_ALIAS}"
            )
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
    coords: Union[Sequence[Tuple[Any, ...]], Mapping[Any, Any], None] = None,
    dims: Sequence = None,
    attrs: dict = None,
) -> xr.DataArray:
    """Loads a tiff stack to an xarray.

    The .tiff format does not retain information on the axes, so these need to
    be manually added with the axes argument. Otherwise, this returns the data
    only as np.ndarray

    Args:
        faddr: Path to file to load.
        coords: The axes describing the data, following the tiff stack order:
        dims: the order of the coordinates provided, considering the data is
        ordered as TZCYXS. If None (default) it infers the order from the order
        of the coords dictionary.
        attrs: dictionary to add as attributes to the xarray.DataArray

    Returns:
        data: an xarray representing the data loaded from the .tiff
        file
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

"""This module contains hdf5 file input/output functions for the sed.io module

"""
from typing import Union

import h5py
import numpy as np
import xarray as xr


def recursive_write_metadata(h5group: h5py.Group, node: dict):
    """Recurses through a python dictionary and writes it into an hdf5 file.

    Args:
        h5group (h5py.Group): hdf5 group element where to store the current dict node
            to.
        node (dict): dictionary node to store

    Raises:
        Warning: warns if elements have been converted into strings for saving.
        ValueError: Raised when elements cannot be saved even as strings.
    """
    for key, item in node.items():
        if isinstance(
            item,
            (
                np.ndarray,
                np.int64,
                np.float64,
                str,
                bytes,
                int,
                float,
                list,
            ),
        ):
            try:
                h5group.create_dataset(key, data=item)
            except TypeError:
                h5group.create_dataset(key, data=str(item))
                print(f"Saved {key} as string.")
        elif isinstance(item, dict):
            group = h5group.create_group(key)
            recursive_write_metadata(group, item)
        else:
            try:
                h5group.create_dataset(key, data=str(item))
                print(f"Saved {key} as string.")
            except BaseException as exc:
                raise ValueError(
                    f"Unknown error occured, cannot save {item} of type {type(item)}.",
                ) from exc


def recursive_parse_metadata(
    node: Union[h5py.Group, h5py.Dataset],
) -> dict:
    """Recurses through an hdf5 file, and parse it into a dictionary.

    Args:
        node (Union[h5py.Group, h5py.Dataset]): hdf5 group or dataset to parse into
            dictionary.

    Returns:
        dict: Dictionary of elements in the hdf5 path contained in node
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
        data (xr.DataArray): input data
        faddr (str): complete file name (including path)
        mode (str, optional): hdf5 read/write mode. Defaults to "w".

    Raises:
        Warning: subfunction warns if elements have been converted into strings for
            saving.
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
        faddr (str): complete file name (including path)
        mode (str, optional): hdf5 read/write mode. Defaults to "r".

    Raises:
        ValueError: Raised if data or axes are not found in the file.

    Returns:
        xr.DataArray: output xarra data
    """
    with h5py.File(faddr, mode) as h5_file:
        # Reading data array
        try:
            data = np.asarray(h5_file["binned"]["BinnedData"])
        except KeyError as exc:
            raise ValueError(
                f"Wrong Data Format, the BinnedData were not found. The error was{exc}.",
            ) from exc

        # Reading the axes
        bin_axes = []
        bin_names = []

        try:
            for axis in h5_file["axes"]:
                bin_axes.append(h5_file["axes"][axis])
                bin_names.append(h5_file["axes"][axis].attrs["name"])
        except KeyError as exc:
            raise ValueError(
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
            for axis in range(len(bin_axes)):
                xarray[bin_names[axis]].attrs["unit"] = h5_file["axes"][f"ax{axis}"].attrs["unit"]
            xarray.attrs["units"] = h5_file["binned"]["BinnedData"].attrs["units"]
            xarray.attrs["long_name"] = h5_file["binned"]["BinnedData"].attrs["long_name"]
        except (KeyError, TypeError):
            pass

        if metadata is not None:
            xarray.attrs["metadata"] = metadata

        return xarray

from typing import Union

import h5py
import numpy as np
import xarray as xr


def recursive_write_metadata(h5group: h5py.Group, node: dict):
    """Recurces through a python dictionary and writes it into an hdf5 file.

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
            except BaseException:
                raise Exception(
                    f"Unknown error occured, cannot save {item} of type {type(item)}.",
                )


def recursive_parse_metadata(
    node: Union[h5py.Group, h5py.Dataset],
) -> dict:
    """Recurces through an hdf5 file, and parse it into a dictionary.

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
        dictionary = node[...]
        try:
            dictionary = dictionary.item()
            if isinstance(dictionary, (bytes, bytearray)):
                dictionary = dictionary.decode()
        except ValueError:
            pass

    return dictionary


def xarray_to_h5(data: xr.DataArray, faddr: str, mode: str = "w"):
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
    with h5py.File(faddr, mode) as h5File:

        print(f"saving data to {faddr}")

        # Saving data, make a single dataset
        dataset = h5File.create_dataset("binned/BinnedData", data=data.data)
        try:
            dataset.attrs["units"] = data.attrs["units"]
            dataset.attrs["long_name"] = data.attrs["long_name"]
        except KeyError:
            pass

        # Saving axes
        axesGroup = h5File.create_group("axes")
        axesNumber = 0
        for binName in data.dims:
            axis = axesGroup.create_dataset(
                f"ax{axesNumber}",
                data=data.coords[binName],
            )
            axis.attrs["name"] = binName
            try:
                axis.attrs["unit"] = data.coords[binName].attrs["unit"]
            except KeyError:
                pass
            axesNumber += 1

        if "metadata" in data.attrs and isinstance(
            data.attrs["metadata"],
            dict,
        ):
            metaGroup = h5File.create_group("metadata")

            recursive_write_metadata(metaGroup, data.attrs["metadata"])

    print("Saving complete!")


def h5_to_xarray(faddr: str, mode: str = "r") -> xr.DataArray:
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
        except KeyError:
            raise Exception(
                "Wrong Data Format, the BinnedData were not found.",
            )

        # Reading the axes
        binAxes = []
        binNames = []

        try:
            for axis in h5_file["axes"]:
                binAxes.append(h5_file["axes"][axis])
                binNames.append(h5_file["axes"][axis].attrs["name"])
        except KeyError:
            raise Exception("Wrong Data Format, the axes were not found.")

        # load metadata
        metadata = None
        if "metadata" in h5_file:
            metadata = recursive_parse_metadata(h5_file["metadata"])

        coords = {}
        for name, vals in zip(binNames, binAxes):
            coords[name] = vals

        xarray = xr.DataArray(data, dims=binNames, coords=coords)

        try:
            for name in binNames:
                xarray[name].attrs["unit"] = h5_file["axes"][axis].attrs[
                    "unit"
                ]
            xarray.attrs["units"] = h5_file["binned"]["BinnedData"].attrs[
                "units"
            ]
            xarray.attrs["long_name"] = h5_file["binned"]["BinnedData"].attrs[
                "long_name"
            ]
        except KeyError:
            pass

        if metadata is not None:
            xarray.attrs["metadata"] = metadata

        return xarray

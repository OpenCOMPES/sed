from pathlib import Path
from typing import Union

import h5py
import numpy as np
import tifffile
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


def array_to_tiff(array: np.ndarray, faddr: Union[Path, str]) -> None:
    """Save array to imageJ compatible tiff stack.

    Args:
        array: array to save. 2D,3D or 4D
        faddr: full path and name of file to save.

    Raise:
        NotImplementedError: if the input array is not 2,3 or 4 dimensional.
    """
    if array.ndim == 2:
        out = np.expand_dims(array, [0, 1, 2, 5])
    elif array.ndim == 3:
        out = np.expand_dims(array, [0, 2, 5])
    elif array.ndim == 4:
        out = np.expand_dims(array, [2, 5])
    else:
        raise NotImplementedError("Only 2-3-4D arrays supported.")

    tifffile.imwrite(faddr, out.astype(np.float32), imagej=True)
    print(f"Successfully saved array with shape {array.shape} to {faddr}")


def xarray_to_tiff(
    data: xr.DataArray,
    faddr: Union[Path, str],
    axis_dict: dict = None,
) -> None:
    """Save an xarray as a  .tiff stack compatible with ImageJ

    Args:
        data: data to be saved. ImageJ likes tiff files with axis order as
        TZCYXS. Therefore, best axis order in input should be: Time, Energy,
        posY, posX. The channels 'C' and 'S' are automatically added and can
        be ignored.
        faddr: full path and name of file to save.
        axis_dict: name pairs for correct axis ordering. Keys should be any of
        T,Z,C,Y,X,S. The Corresponding value will be searched among the
        dimensions of the xarray, and placed in the right order for imagej
        stacks metadata.

    Raise:
        AttributeError: if more than one axis corresponds to a single dimension
    """

    assert isinstance(data, xr.DataArray), "Data must be an xarray.DataArray"
    dims_to_add = {"C": 1, "S": 1}
    dims_order = []

    if axis_dict is None:
        axis_dict = {
            "T": ["delayStage", "pumpProbeTime", "time"],
            "Z": ["dldTime", "energy"],
            "C": ["C"],
            "Y": ["dldPosY", "ky"],
            "X": ["dldPosX", "kx"],
            "S": ["S"],
        }
    else:
        for key in ["T", "Z", "C", "Y", "X", "S"]:
            if key not in axis_dict.keys():
                axis_dict[key] = key

    # Sort the dimensions in the correct order, and fill with one-point dimensions
    # the missing axes.
    for key in ["T", "Z", "C", "Y", "X", "S"]:
        axis_name_list = [name for name in axis_dict[key] if name in data.dims]
        if len(axis_name_list) > 1:
            raise AttributeError(f"Too many dimensions for {key} axis.")
        elif len(axis_name_list) == 1:
            dims_order.append(*axis_name_list)
        else:
            dims_to_add[key] = 1
            dims_order.append(key)

    xres = data.expand_dims(dims_to_add)
    xres = xres.transpose(*dims_order)
    if ".tif" not in faddr:
        faddr += ".tif"
    tifffile.imwrite(faddr, xres.values.astype(np.float32), imagej=True)

    # resolution=(1./2.6755, 1./2.6755),metadata={'spacing': 3.947368, 'unit': 'um'})
    print(f"Successfully saved {faddr}")


def to_tiff(
    data: Union[np.ndarray, xr.DataArray],
    faddr: Union[Path, str],
    axis_dict: dict = None,
) -> None:
    """Save an array to imagej tiff sack compatible with ImageJ.

    Args:
        data: data to be saved. ImageJ likes tiff files with axis order as
        TZCYXS. Therefore, best axis order in input should be: Time, Energy,
        posY, posX. The channels 'C' and 'S' are automatically added and can
        be ignored.
        faddr: full path and name of file to save.
        axis_dict: name pairs for correct axis ordering. Keys should be any of
        T,Z,C,Y,X,S. The Corresponding value will be searched among the
        dimensions of the xarray, and placed in the right order for imagej
        stacks metadata.

    Raises:
        TypeError: when an incompatible data format is provided.
    """
    try:
        if isinstance(data, xr.DataArray):
            xarray_to_tiff(data, faddr, axis_dict=axis_dict)
        elif isinstance(data, np.ndarray):
            array_to_tiff(data, faddr)
        else:
            array_to_tiff(np.array(data), faddr)
    except Exception:
        raise TypeError("Input data must be a numpy array or xarray DataArray")

from pathlib import Path
from typing import Sequence
from typing import Union

import h5py
import numpy as np
import tifffile
import xarray as xr


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
            except BaseException:
                raise Exception(
                    f"Unknown error occured, cannot save {item} of type {type(item)}.",
                )


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
        dictionary = node[...]
        try:
            dictionary = dictionary.item()
            if isinstance(dictionary, (bytes, bytearray)):
                dictionary = dictionary.decode()
        except ValueError:
            pass

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


def to_tiff(
    data: Union[xr.DataArray, np.ndarray],
    faddr: Union[Path, str],
    axis_dict: dict = None,
    ret_axes_order: bool = False,
) -> None:
    """Save an array as a  .tiff stack compatible with ImageJ

    Args:
        data: data to be saved. If a np.ndarray, the order is retained. If it
        is an xarray.DataArray, the order is inferred from axis_dict instead.
         ImageJ likes tiff files with axis order as
        TZCYXS. Therefore, best axis order in input should be: Time, Energy,
        posY, posX. The channels 'C' and 'S' are automatically added and can
        be ignored.
        faddr: full path and name of file to save.
        axis_dict: name pairs for correct axis ordering. Keys should be any of
        T,Z,C,Y,X,S. The Corresponding value will be searched among the
        dimensions of the xarray, and placed in the right order for imagej
        stacks metadata. If None it tries to guess the order from the name of
        the axes. Defaults to None

    Raise:
        AttributeError: if more than one axis corresponds to a single dimension
        NotImplementedError: if data is not 2,3 or 4 dimensional
        TypeError: if data is not a np.ndarray or an xarray.DataArray
    """
    _imagej_axes_order = ["T", "Z", "C", "Y", "X", "S"]

    if isinstance(data, np.ndarray):
        dim_expansions = {2: [0, 1, 2, 5], 3: [0, 2, 5], 4: [2, 5]}
        dims_order = data.dims
        try:
            out = np.expand_dims(data, dim_expansions[data.ndim])
        except KeyError:
            raise NotImplementedError(
                f"Only 2-3-4D arrays supported when data is a {type(data)}",
            )

    elif isinstance(data, xr.DataArray):
        dims_to_add = {"C": 1, "S": 1}
        dims_order = []

        if axis_dict is None:
            axis_dict = {
                "T": [
                    "delayStage",
                    "pumpProbeTime",
                    "time",
                    "delay",
                ],
                "Z": [
                    "dldTime",
                    "t",
                    "energy",
                    "e",
                    "binding_energy",
                    "energies",
                    "binding_energies",
                ],
                "C": ["C"],
                "Y": ["dldPosY", "ky", "y", "ypos", "Y"],
                "X": ["dldPosX", "kx", "x", "xpos", "X"],
                "S": ["S"],
            }
        else:
            for key in _imagej_axes_order:
                if key not in axis_dict.keys():
                    axis_dict[key] = key

        # Sort the dimensions in the correct order, and fill with one-point dimensions
        # the missing axes.
        for key in _imagej_axes_order:
            axis_name_list = [
                name for name in axis_dict[key] if name in data.dims
            ]
            if len(axis_name_list) > 1:
                raise AttributeError(f"Too many dimensions for {key} axis.")
            elif len(axis_name_list) == 1:
                dims_order.append(*axis_name_list)
            else:
                dims_to_add[key] = 1
                dims_order.append(key)

        out = data.expand_dims(dims_to_add)
        out = out.transpose(*dims_order).values
    else:
        raise TypeError(f"Cannot handle data of type {data.type}")

    faddr = Path(faddr).with_suffix(".tiff")

    tifffile.imwrite(faddr, out.astype(np.float32), imagej=True)
    # clean up the temporary axes names
    for ax in _imagej_axes_order:
        if ax not in data.dims:
            try:
                dims_order.remove(ax)
            except ValueError:
                pass

    print(f"Successfully saved {faddr}\n Axes order: {dims_order}")
    if ret_axes_order:
        return dims_order


def load_tiff(
    faddr: Union[str, Path],
    coords: Union[Sequence[str], dict] = None,
    dims: Sequence = None,
    attrs: dict = None,
) -> Union[np.ndarray, xr.DataArray]:
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
        data: a np.array or xarray representing the data loaded from the .tiff
        file
    """
    data = tifffile.imread(faddr).squeeze()
    if coords is None:
        return data
    else:
        if dims is None:
            dims = list(coords.keys())
        assert data.ndim == len(coords) == len(dims), (
            f"Data dimension {data.ndim} must coincide number of coordinates"
            f"{len(coords)} and dimensions {len(dims)} provided,"
        )
        return xr.DataArray(data=data, coords=coords, dims=dims, attrs=attrs)

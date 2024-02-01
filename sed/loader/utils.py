"""Utilities for loaders
"""
from glob import glob
from typing import cast
from typing import List
from typing import Sequence
from typing import Union

import dask.dataframe
import numpy as np
import pandas as pd
from h5py import File
from h5py import Group
from natsort import natsorted


def gather_files(
    folder: str,
    extension: str,
    f_start: int = None,
    f_end: int = None,
    f_step: int = 1,
    file_sorting: bool = True,
) -> List[str]:
    """Collects and sorts files with specified extension from a given folder.

    Args:
        folder (str): The folder to search
        extension (str):  File extension used for glob.glob().
        f_start (int, optional): Start file id used to construct a file selector.
            Defaults to None.
        f_end (int, optional): End file id used to construct a file selector.
            Defaults to None.
        f_step (int, optional): Step of file id incrementation, used to construct
            a file selector. Defaults to 1.
        file_sorting (bool, optional): Option to sort the files by their names.
            Defaults to True.

    Returns:
        List[str]: List of collected file names.
    """
    try:
        files = glob(folder + "/*." + extension)

        if file_sorting:
            files = cast(List[str], natsorted(files))

        if f_start is not None and f_end is not None:
            files = files[slice(f_start, f_end, f_step)]

    except FileNotFoundError:
        print("No legitimate folder address is specified for file retrieval!")
        raise

    return files


def parse_h5_keys(h5_file: File, prefix: str = "") -> List[str]:
    """Helper method which parses the channels present in the h5 file
    Args:
        h5_file (h5py.File): The H5 file object.
        prefix (str, optional): The prefix for the channel names.
        Defaults to an empty string.

    Returns:
        List[str]: A list of channel names in the H5 file.

    Raises:
        Exception: If an error occurs while parsing the keys.
    """

    # Initialize an empty list to store the channels
    file_channel_list = []

    # Iterate over the keys in the H5 file
    for key in h5_file.keys():
        try:
            # Check if the object corresponding to the key is a group
            if isinstance(h5_file[key], Group):
                # If it's a group, recursively call the function on the group object
                # and append the returned channels to the file_channel_list
                file_channel_list.extend(
                    parse_h5_keys(h5_file[key], prefix=prefix + "/" + key),
                )
            else:
                # If it's not a group (i.e., it's a dataset), append the key
                # to the file_channel_list
                file_channel_list.append(prefix + "/" + key)
        except KeyError as exception:
            # If an exception occurs, raise a new exception with an error message
            raise KeyError(
                f"Error parsing key: {prefix}/{key}",
            ) from exception

    # Return the list of channels
    return file_channel_list


def split_channel_bitwise(
    df: dask.dataframe.DataFrame,
    input_column: str,
    output_columns: Sequence[str],
    bit_mask: int,
    overwrite: bool = False,
    types: Sequence[type] = None,
) -> dask.dataframe.DataFrame:
    """Splits a channel into two channels bitwise.

    This function splits a channel into two channels by separating the first n bits from
    the remaining bits. The first n bits are stored in the first output column, the
    remaining bits are stored in the second output column.

    Args:
        df (dask.dataframe.DataFrame): Dataframe to use.
        input_column (str): Name of the column to split.
        output_columns (Sequence[str]): Names of the columns to create.
        bit_mask (int): Bit mask to use for splitting.
        overwrite (bool, optional): Whether to overwrite existing columns.
            Defaults to False.
        types (Sequence[type], optional): Types of the new columns.

    Returns:
        dask.dataframe.DataFrame: Dataframe with the new columns.
    """
    if len(output_columns) != 2:
        raise ValueError("Exactly two output columns must be given.")
    if input_column not in df.columns:
        raise KeyError(f"Column {input_column} not in dataframe.")
    if output_columns[0] in df.columns and not overwrite:
        raise KeyError(f"Column {output_columns[0]} already in dataframe.")
    if output_columns[1] in df.columns and not overwrite:
        raise KeyError(f"Column {output_columns[1]} already in dataframe.")
    if bit_mask < 0 or not isinstance(bit_mask, int):
        raise ValueError("bit_mask must be a positive. integer")
    if types is None:
        types = [np.int8 if bit_mask < 8 else np.int16, np.int32]
    elif len(types) != 2:
        raise ValueError("Exactly two types must be given.")
    elif not all(isinstance(t, type) for t in types):
        raise ValueError("types must be a sequence of types.")
    df[output_columns[0]] = (df[input_column] % 2**bit_mask).astype(types[0])
    df[output_columns[1]] = (df[input_column] // 2**bit_mask).astype(types[1])
    return df


def split_dld_time_from_sector_id(
    df: Union[pd.DataFrame, dask.dataframe.DataFrame],
    tof_column: str = None,
    sector_id_column: str = None,
    sector_id_reserved_bits: int = None,
    config: dict = None,
) -> Union[pd.DataFrame, dask.dataframe.DataFrame]:
    """Converts the 8s time in steps to time in steps and sectorID.

    The 8s detector encodes the dldSectorID in the 3 least significant bits of the
    dldTimeSteps channel.

    Args:
        df (Union[pd.DataFrame, dask.dataframe.DataFrame]): Dataframe to use.
        tof_column (str, optional): Name of the column containing the
            time-of-flight steps. Defaults to config["dataframe"]["tof_column"].
        sector_id_column (str, optional): Name of the column containing the
            sectorID. Defaults to config["dataframe"]["sector_id_column"].
        sector_id_reserved_bits (int, optional): Number of bits reserved for the
        config (dict, optional): Configuration dictionary. Defaults to None.

    Returns:
        Union[pd.DataFrame, dask.dataframe.DataFrame]: Dataframe with the new columns.
    """
    if tof_column is None:
        if config is None:
            raise ValueError("Either tof_column or config must be given.")
        tof_column = config["dataframe"]["tof_column"]
    if sector_id_column is None:
        if config is None:
            raise ValueError("Either sector_id_column or config must be given.")
        sector_id_column = config["dataframe"]["sector_id_column"]
    if sector_id_reserved_bits is None:
        if config is None:
            raise ValueError("Either sector_id_reserved_bits or config must be given.")
        sector_id_reserved_bits = config["dataframe"].get("sector_id_reserved_bits", None)
        if sector_id_reserved_bits is None:
            raise ValueError('No value for "sector_id_reserved_bits" found in config.')

    if sector_id_column in df.columns:
        raise ValueError(
            f"Column {sector_id_column} already in dataframe. This function is not idempotent.",
        )
    df = split_channel_bitwise(
        df=df,
        input_column=tof_column,
        output_columns=[sector_id_column, tof_column],
        bit_mask=sector_id_reserved_bits,
        overwrite=True,
        types=[np.int8, np.int32],
    )
    return df

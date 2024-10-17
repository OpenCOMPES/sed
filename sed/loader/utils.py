"""Utilities for loaders
"""
from __future__ import annotations

from collections.abc import Sequence
from glob import glob
from pathlib import Path
from typing import cast

import dask.dataframe
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
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
) -> list[str]:
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
        list[str]: List of collected file names.
    """
    try:
        files = glob(folder + "/*." + extension)

        if file_sorting:
            files = cast(list[str], natsorted(files))

        if f_start is not None and f_end is not None:
            files = files[slice(f_start, f_end, f_step)]

    except FileNotFoundError:
        print("No legitimate folder address is specified for file retrieval!")
        raise

    return files


def parse_h5_keys(h5_file: File, prefix: str = "") -> list[str]:
    """Helper method which parses the channels present in the h5 file
    Args:
        h5_file (h5py.File): The H5 file object.
        prefix (str, optional): The prefix for the channel names.
        Defaults to an empty string.

    Returns:
        list[str]: A list of channel names in the H5 file.

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
    df: pd.DataFrame | dask.dataframe.DataFrame,
    tof_column: str = None,
    sector_id_column: str = None,
    sector_id_reserved_bits: int = None,
    config: dict = None,
) -> tuple[pd.DataFrame | dask.dataframe.DataFrame, dict]:
    """Converts the 8s time in steps to time in steps and sectorID.

    The 8s detector encodes the dldSectorID in the 3 least significant bits of the
    dldTimeSteps channel.

    Args:
        df (pd.DataFrame | dask.dataframe.DataFrame): Dataframe to use.
        tof_column (str, optional): Name of the column containing the
            time-of-flight steps. Defaults to config["dataframe"]["columns"]["tof"].
        sector_id_column (str, optional): Name of the column containing the
            sectorID. Defaults to config["dataframe"]["columns"]["sector_id"].
        sector_id_reserved_bits (int, optional): Number of bits reserved for the
        config (dict, optional): Dataframe configuration dictionary. Defaults to None.

    Returns:
        pd.DataFrame | dask.dataframe.DataFrame: Dataframe with the new columns.
    """
    if tof_column is None:
        if config is None:
            raise ValueError("Either tof_column or config must be given.")
        tof_column = config["columns"]["tof"]
    if sector_id_column is None:
        if config is None:
            raise ValueError("Either sector_id_column or config must be given.")
        sector_id_column = config["columns"]["sector_id"]
    if sector_id_reserved_bits is None:
        if config is None:
            raise ValueError("Either sector_id_reserved_bits or config must be given.")
        sector_id_reserved_bits = config.get("sector_id_reserved_bits", None)
        if sector_id_reserved_bits is None:
            raise ValueError('No value for "sector_id_reserved_bits" found in config.')

    if sector_id_column in df.columns:
        metadata = {"applied": False, "reason": f"Column {sector_id_column} already in dataframe"}
    else:
        # Split the time-of-flight column into sector ID and time-of-flight steps
        df = split_channel_bitwise(
            df=df,
            input_column=tof_column,
            output_columns=[sector_id_column, tof_column],
            bit_mask=sector_id_reserved_bits,
            overwrite=True,
            types=[np.int8, np.int32],
        )
        metadata = {
            "applied": True,
            "tof_column": tof_column,
            "sector_id_column": sector_id_column,
            "sector_id_reserved_bits": sector_id_reserved_bits,
        }

    return df, {"split_dld_time_from_sector_id": metadata}


def get_stats(meta: pq.FileMetaData) -> dict:
    """
    Extracts the minimum and maximum of all columns from the metadata of a Parquet file.

    Args:
        meta (pq.FileMetaData): The metadata of the Parquet file.

    Returns:
        Tuple[int, int]: The minimum and maximum timestamps.
    """
    min_max = {}
    for idx, name in enumerate(meta.schema.names):
        col = []
        for i in range(meta.num_row_groups):
            stats = meta.row_group(i).column(idx).statistics
            if stats is not None:
                if stats.min is not None:
                    col.append(stats.min)
                if stats.max is not None:
                    col.append(stats.max)
        if col:
            min_max[name] = {"min": min(col), "max": max(col)}
    return min_max


def get_parquet_metadata(file_paths: list[Path]) -> dict[str, dict]:
    """
    Extracts and organizes metadata from a list of Parquet files.

    For each file, the function reads the metadata, adds the filename,
    and extracts the minimum and maximum timestamps.
    "row_groups" entry is removed from FileMetaData.

    Args:
        file_paths (list[Path]): A list of paths to the Parquet files.

    Returns:
        dict[str, dict]: A dictionary file index as key and the values as metadata of each file.
    """
    organized_metadata = {}
    for i, file_path in enumerate(file_paths):
        # Read the metadata for the file
        file_meta: pq.FileMetaData = pq.read_metadata(file_path)
        # Convert the metadata to a dictionary
        metadata_dict = file_meta.to_dict()
        # Add the filename to the metadata dictionary
        metadata_dict["filename"] = str(file_path.name)

        # Get column min and max values
        metadata_dict["columns"] = get_stats(file_meta)

        # Remove "row_groups" as they contain a lot of info that is not needed
        metadata_dict.pop("row_groups", None)

        # Add the metadata dictionary to the organized_metadata dictionary
        organized_metadata[str(i)] = metadata_dict

    return organized_metadata

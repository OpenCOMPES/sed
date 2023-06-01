"""Utilities for flash loader"""
import os
from pathlib import Path
from typing import List


def initialize_paths(df_config):
    """
    Initializes the paths
    """
    # Prases to locate the raw beamtime directory from config file
    if "paths" in df_config:
        paths = df_config["paths"]
        if "data_raw_dir" in paths:
            data_raw_dir = Path(paths["data_raw_dir"])
        if "data_parquet_dir" in paths:
            data_parquet_dir = Path(paths["data_parquet_dir"])
            if not data_parquet_dir.exists():
                os.mkdir(data_parquet_dir)

    if "paths" not in df_config:
        if not {"ubid_offset", "daq"}.issubset(df_config.keys()):
            raise ValueError(
                "One of the values from ubid_offset or daq is missing. \
                        These are necessary.",
            )

        if not {"beamtime_id", "year"}.issubset(df_config.keys()):
            raise ValueError(
                "The beamtime_id and year or data_raw_dir is required.",
            )

        beamtime_id = df_config["beamtime_id"]
        year = df_config["year"]
        beamtime_dir = Path(
            f"/asap3/flash/gpfs/pg2/{year}/data/{beamtime_id}/",
        )

        if {"instrument"}.issubset(df_config.keys()):
            instrument = df_config["instrument"]
            if instrument == "wespe":
                beamtime_dir = Path(
                    f"/asap3/fs-flash-o/gpfs/{instrument}/{year}/data/{beamtime_id}/",
                )

        daq = df_config["daq"]

        # Use os walk to reach the raw data directory
        data_raw_dir = []
        for root, dirs, files in os.walk(beamtime_dir.joinpath("raw/")):
            for dir_name in dirs:
                if dir_name.startswith("express-") or dir_name.startswith(
                    "online-",
                ):
                    data_raw_dir.append(Path(root, dir_name, daq))
                elif dir_name == daq.upper():
                    data_raw_dir.append(Path(root, dir_name))

        if not data_raw_dir:
            raise FileNotFoundError("Raw data directories not found.")

        parquet_path = "processed/parquet"
        data_parquet_dir = beamtime_dir.joinpath(parquet_path)

        if not data_parquet_dir.exists():
            os.mkdir(data_parquet_dir)

    return data_raw_dir, data_parquet_dir


def gather_flash_files(
    run_number: int,
    daq: str,
    raw_data_dirs: List[str],
    extension: str = "h5",
) -> List[Path]:
    """Returns a list of filenames for a given run located in the specified directory
    for the specified data acquisition (daq).

    Args:
        run_number (int): The number of the run.
        daq (str): The data acquisition identifier.
        raw_data_dir (str): The directory where the raw data is located.
        extension (str, optional): The file extension. Defaults to "h5".

    Returns:
        List[Path]: A list of Path objects representing the collected file names.

    Raises:
        FileNotFoundError: If no files are found for the given run in the directory.
    """
    # Define the stream name prefixes based on the data acquisition identifier
    stream_name_prefixes = {
        "pbd": "GMD_DATA_gmd_data",
        "pbd2": "FL2PhotDiag_pbd2_gmd_data",
        "fl1user1": "FLASH1_USER1_stream_2",
        "fl1user2": "FLASH1_USER2_stream_2",
        "fl1user3": "FLASH1_USER3_stream_2",
        "fl2user1": "FLASH2_USER1_stream_2",
        "fl2user2": "FLASH2_USER2_stream_2",
    }

    # Generate the file patterns to search for in the directory
    file_pattern = (
        f"{stream_name_prefixes[daq]}_run{run_number}_*." + extension
    )

    files = []
    raw_data_dirs = (
        raw_data_dirs if isinstance(raw_data_dirs, list) else [raw_data_dirs]
    )
    # search through all directories
    for raw_data_dir in raw_data_dirs:
        # Use pathlib to search for matching files in each directory
        files.extend(
            sorted(
                Path(raw_data_dir).glob(file_pattern),
                key=lambda filename: str(filename).rsplit("_", maxsplit=1)[-1],
            ),
        )

    # Check if any files are found
    if not files:
        raise FileNotFoundError(
            f"No files found for run {run_number} in directory {raw_data_dir}",
        )

    # Return the list of found files
    return files

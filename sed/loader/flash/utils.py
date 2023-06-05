"""Utilities for flash loader"""
import os
from importlib.util import find_spec
from pathlib import Path
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple

from sed.config.settings import load_config

# load identifiers
package_dir = os.path.dirname(find_spec("sed").origin)
identifiers_path = f"{package_dir}/loader/flash/identifiers.json"
identifiers = load_config(identifiers_path)


def initialize_paths(df_config: Dict) -> Tuple[Path, Path]:
    """
    Initializes the paths based on the configuration.

    Args:
        df_config (Dict): A dictionary containing the configuration.

    Returns:
        Tuple[Path, Path]: A tuple containing the raw data directory path
        and the parquet data directory path.

    Raises:
        ValueError: If required values are missing from the configuration.
        FileNotFoundError: If the raw data directories are not found.
    """
    # Parses to locate the raw beamtime directory from config file
    if "paths" in df_config:
        paths = df_config["paths"]
        if "data_raw_dir" in paths:
            data_raw_dir = Path(paths["data_raw_dir"])
        if "data_parquet_dir" in paths:
            data_parquet_dir = Path(paths["data_parquet_dir"])
            if not data_parquet_dir.exists():
                os.mkdir(data_parquet_dir)

    else:
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
        daq = df_config["daq"]

        beamtime_dir = Path(
            identifiers["beamtime_dir"][df_config["instrument"]],
        )
        beamtime_dir = beamtime_dir.joinpath(f"{year}/data/{beamtime_id}/")

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
    raw_data_dirs: Sequence[Path],
    extension: str = "h5",
) -> List[Path]:
    """Returns a list of filenames for a given run located in the specified directory
    for the specified data acquisition (daq).

    Args:
        run_number (int): The number of the run.
        daq (str): The data acquisition identifier.
        raw_data_dir (Sequence[Path]): The directory where the raw data is located.
        extension (str, optional): The file extension. Defaults to "h5".

    Returns:
        List[Path]: A list of Path objects representing the collected file names.

    Raises:
        FileNotFoundError: If no files are found for the given run in the directory.
    """
    # Define the stream name prefixes based on the data acquisition identifier
    stream_name_prefixes = identifiers["stream_name_prefixes"]

    # Generate the file patterns to search for in the directory
    file_pattern = (
        f"{stream_name_prefixes[daq]}_run{run_number}_*." + extension
    )

    files = []
    raw_data_dirs = list(raw_data_dirs)
    # search through all directories
    for raw_data_dir in raw_data_dirs:
        # Use pathlib to search for matching files in each directory
        files.extend(
            sorted(
                raw_data_dir.glob(file_pattern),
                key=lambda filename: str(filename).rsplit("_", maxsplit=1)[-1],
            ),
        )

    # Check if any files are found
    if not files:
        raise FileNotFoundError(
            f"No files found for run {run_number} in directory {str(raw_data_dirs)}",
        )

    # Return the list of found files
    return files

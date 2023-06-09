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

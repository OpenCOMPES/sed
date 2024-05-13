"""This module provides functions to fetch and load datasets."""
from __future__ import annotations

import json
import os
import shutil
import zipfile

import requests

from sed.core.config import setup_logging
from sed.core.config import user_dirs

# Configure logging
logger = setup_logging()

# check if datasets.json exists in user_config_dir
json_path = os.path.join(user_dirs.user_config_dir, "datasets", "datasets.json")
if not os.path.exists(json_path):
    shutil.copy(
        os.path.join(os.path.dirname(__file__), "datasets.json"),
        json_path,
    )
DATASETS = json.load(open(json_path))


def available_datasets() -> list:
    """
    Returns a list of available datasets.

    Returns:
        list: List of available datasets.
    """
    return list(DATASETS.keys())


def get_file_list(directory: str, ignore_zip: bool = True) -> dict:
    """
    Returns a dictionary containing lists of files in each subdirectory
    and files in the main directory.

    Args:
        directory (str): Path to the directory.
        ignore_zip (bool): Whether to ignore ZIP files. Default is True.

    Returns:
        dict: Dictionary containing lists of files in each subdirectory and files
                in the main directory.
    """
    result = {}
    main_dir_files = []

    # List all files and directories in the given directory
    all_files_and_dirs = os.listdir(directory)

    # Filter out hidden files and directories
    visible_files_and_dirs = [item for item in all_files_and_dirs if not item.startswith(".")]

    # Organize files into dictionary structure
    for item in visible_files_and_dirs:
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            result[item] = [file for file in os.listdir(item_path) if not file.startswith(".")]
        else:
            main_dir_files.append(item)

    # Include files in the main directory
    result["main_directory"] = main_dir_files

    # remove zip files if ignore_zip is True
    if ignore_zip:
        for key in list(result.keys()):
            result[key] = [file for file in result[key] if not file.endswith(".zip")]

    return result


def download_data(data_name: str, data_path: str, data_url: str) -> None:
    """
    Downloads data from the specified URL.

    Args:
        data_name (str): Name of the data.
        data_path (str): Path where the data should be stored.
        data_url (str): URL of the data.
    """
    zip_file_path = os.path.join(data_path, f"{data_name}.zip")

    # Check if data already exists
    if not os.path.exists(zip_file_path):
        logger.info(f"Downloading {data_name} data...")
        response = requests.get(data_url)
        with open(zip_file_path, "wb") as f:
            f.write(response.content)
        logger.info("Download complete.")
    else:
        logger.info(f"{data_name} data is already downloaded.")


def extract_data(data_name: str, data_path: str, subdirs: list) -> None:
    """
    Extracts data from a ZIP file.

    Args:
        data_name (str): Name of the data.
        data_path (str): Path where the data should be stored.
        subdirs (list): List of subdirectories.
    """
    zip_file_path = os.path.join(data_path, f"{data_name}.zip")
    # Set Extract data flag to true, if not already extracted
    extract = True
    for subdir in subdirs:
        if os.path.isdir(os.path.join(data_path, subdir)):
            extract = False

    if extract:
        logger.info(f"Extracting {data_name} data...")
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(data_path)
        logger.info("Extraction complete.")
    else:
        logger.info(f"{data_name} data is already extracted.")


def rearrange_data(data_path: str, subdirs: list, rearrange_files: bool) -> None:
    """
    Moves files to the main directory if specified.

    Args:
        data_path (str): Path where the data should be stored.
        subdirs (list): List of subdirectories.
        rearrange_files (bool): Whether to rearrange files.
    """
    if rearrange_files:
        for subdir in subdirs:
            source_path = os.path.join(data_path, subdir)
            if os.path.isdir(source_path):
                logger.info(f"Rearranging files...")
                for root, dirs, files in os.walk(source_path):
                    for file in files:
                        shutil.move(os.path.join(root, file), data_path)
                logger.info("File movement complete.")
                shutil.rmtree(source_path)
            else:
                logger.error(f"Subdirectory {subdir} not found.")
        logger.info("Rearranging complete.")


def load_dataset(data_name: str, data_path: str = None) -> str | tuple[str, list[str]]:
    """
    Fetches the specified data and extracts it to the given data path.

    Args:
        data_name (str): Name of the data to fetch.
        data_path (str): Path where the data should be stored. Default is the current directory.

    Returns:
        str | tuple[str, list[str]]: Path to the dataset or a tuple containing the path to the
        dataset and subdirectories.
    """
    # Check if the data is available
    if data_name not in DATASETS:
        error_message = f"Data '{data_name}' is not available for fetching.\
                Available datasets are: {available_datasets()}"
        logger.error(error_message)
        raise ValueError(error_message)

    dataset = DATASETS.get(data_name)
    subdirs = dataset.get("subdirs", [])
    url = dataset.get("url")
    rearrange_files = dataset.get("rearrange_files", False)
    existing_data_path = dataset.get("data_path", None)
    existing_files = dataset.get("files", {})

    # Notify the user if data might already exist
    if existing_data_path and data_path and existing_data_path != data_path:
        logger.info(
            f'{data_name} data might already exists at "{existing_data_path}", '
            "unless deleted manually.",
        )

    # Set data path if not provided
    if data_path is None:
        if existing_data_path:
            data_path = existing_data_path
        else:
            # create a new dir in user_dirs.user_data_dir with data_name
            data_path = os.path.join(user_dirs.user_data_dir, "datasets", data_name)
        logger.info(f'Data path not provided. Using path: "{data_path}"')

        if not os.path.exists(data_path):
            os.makedirs(data_path)

    files_in_dir = get_file_list(data_path)

    # if existing_files is same as files_in_dir, then don't download/extract data
    if existing_files == files_in_dir:
        logger.info(f"{data_name} data is already downloaded and extracted.")
    else:
        download_data(data_name, data_path, url)
        extract_data(data_name, data_path, subdirs)
        rearrange_data(data_path, subdirs, rearrange_files)

        # Update datasets JSON
        dataset["files"] = get_file_list(data_path)
        dataset["data_path"] = data_path
        with open(json_path, "w") as f:
            json.dump(DATASETS, f, indent=4)

    # Return subdirectory paths if present
    if subdirs and not rearrange_files:
        return data_path, [os.path.join(data_path, subdir) for subdir in subdirs]
    else:
        return data_path

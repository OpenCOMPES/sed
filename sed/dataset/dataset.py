"""This module provides functions to fetch and load datasets."""
from __future__ import annotations

import os
import shutil
import zipfile

import requests

from sed.core.config import load_config
from sed.core.config import save_config
from sed.core.logging import setup_logging
from sed.core.user_dirs import USER_CONFIG_PATH
from sed.core.user_dirs import USER_DATA_PATH

# Configure logging
logger = setup_logging(__name__)

DATASETS_FILENAME = "datasets.json"
# Paths for user configuration and data directories
USER_CONFIG_DATASETS_DIR = USER_CONFIG_PATH / "datasets"
USER_CONFIG_DATASETS_DIR.mkdir(parents=True, exist_ok=True)

USER_DATASETS_DIR = USER_DATA_PATH / "datasets"
USER_DATASETS_DIR.mkdir(parents=True, exist_ok=True)

# Paths for the datasets JSON file
USER_JSON_PATH = USER_DATASETS_DIR.joinpath(DATASETS_FILENAME)
MODULE_JSON_PATH = os.path.join(os.path.dirname(__file__), DATASETS_FILENAME)


def load_datasets_dict() -> dict:
    """
    Loads the datasets configuration dictionary from the user's datasets JSON file.

    If the file does not exist, it copies the default datasets JSON file from the module
    directory to the user's datasets directory.

    Returns:
        dict: The datasets dict loaded from user's datasets JSON file.
    """
    # check if datasets.json exists in user_config_dir
    if not os.path.exists(USER_JSON_PATH):
        module_json = os.path.join(os.path.dirname(__file__), DATASETS_FILENAME)
        shutil.copy(module_json, USER_JSON_PATH)
    datasets = load_config(str(USER_JSON_PATH))
    return datasets


def available_datasets() -> list:
    """
    Returns a list of available datasets.

    Returns:
        list: List of available datasets.
    """
    datasets = load_datasets_dict()
    return list(datasets)


def check_dataset_availability(data_name: str) -> dict:
    """
    Checks if the specified dataset is available in the predefined list of datasets.

    Args:
        data_name (str): The name of the dataset to check.

    Returns:
        dict: The dataset information if available.

    Raises:
        ValueError: If the dataset is not found in the predefined list.
    """
    datasets = load_datasets_dict()
    if data_name not in datasets:
        error_message = f"Data '{data_name}' is not available for fetching.\
            Available datasets are: {available_datasets()}"
        logger.error(error_message)
        raise ValueError(error_message)
    return datasets.get(data_name)


def set_data_path(data_name: str, data_path: str, existing_data_path: str) -> str:
    """
    Determines and sets the data path for a dataset. If a data path is not provided,
    it uses the existing data path or creates a new one. It also notifies the user
    if the specified data path differs from an existing data path.

    Args:
        data_name (str): The name of the dataset.
        data_path (str, optional): The desired path where the dataset should be stored.
        existing_data_path (str, optional): The path where the dataset currently exists.

    Returns:
        str: The final data path for the dataset.
    """
    # Notify the user if data might already exist
    if existing_data_path and data_path and existing_data_path != data_path:
        logger.warning(
            f'{data_name} data might already exists at "{existing_data_path}", '
            "unless deleted manually.",
        )

    # Set data path if not provided
    if data_path is None:
        data_path = existing_data_path or str(
            USER_DATA_PATH.joinpath(
                "datasets",
                data_name,
            ),
        )
        path_source = "existing" if existing_data_path else "default"
        logger.info(f'Using {path_source} data path for "{data_name}": "{data_path}"')

        if not os.path.exists(data_path):
            os.makedirs(data_path)
    return data_path


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


def download_data(data_name: str, data_path: str, data_url: str) -> bool:
    """
    Downloads data from the specified URL.

    Args:
        data_name (str): Name of the data.
        data_path (str): Path where the data should be stored.
        data_url (str): URL of the data.

    Returns:
        bool: True if the data was downloaded successfully,
                False if the data is already downloaded.
    """
    zip_file_path = os.path.join(data_path, f"{data_name}.zip")

    # Check if data already exists
    if not os.path.exists(zip_file_path):
        logger.info(f"Downloading {data_name} data...")
        response = requests.get(data_url)
        with open(zip_file_path, "wb") as f:
            f.write(response.content)
        logger.info("Download complete.")
        downloaded = True
    else:
        logger.info(f"{data_name} data is already downloaded.")
        downloaded = False

    return downloaded


def extract_data(data_name: str, data_path: str, subdirs: list) -> bool:
    """
    Extracts data from a ZIP file.

    Args:
        data_name (str): Name of the data.
        data_path (str): Path where the data should be stored.
        subdirs (list): List of subdirectories.

    Returns:
        bool: True if the data was extracted successfully,
                False if the data is already extracted.
    """
    zip_file_path = os.path.join(data_path, f"{data_name}.zip")
    # Very basic check. Looks if subdirs are empty
    extract = True
    for subdir in subdirs:
        if os.path.isdir(os.path.join(data_path, subdir)):
            extract = False

    if extract:
        logger.info(f"Extracting {data_name} data...")
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(data_path)
        logger.info("Extraction complete.")
        extracted = True
    else:
        logger.info(f"{data_name} data is already extracted.")
        extracted = False

    return extracted


def rearrange_data(data_path: str, subdirs: list) -> None:
    """
    Moves files to the main directory if specified.

    Args:
        data_path (str): Path where the data should be stored.
        subdirs (list): List of subdirectories.
    """
    for subdir in subdirs:
        source_path = os.path.join(data_path, subdir)
        if os.path.isdir(source_path):
            logger.info(f"Rearranging files.")
            for root, dirs, files in os.walk(source_path):
                for file in files:
                    shutil.move(os.path.join(root, file), data_path)
            logger.info("File movement complete.")
            shutil.rmtree(source_path)
        else:
            error_message = f"Subdirectory {subdir} not found."
            logger.error(error_message)
            raise FileNotFoundError(error_message)
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

    dataset = check_dataset_availability(data_name)
    subdirs = dataset.get("subdirs", [])
    rearrange_files = dataset.get("rearrange_files", False)
    if rearrange_files and not subdirs:
        err = f"Rearrange_files is set to True but no subdirectories are defined for {data_name}."
        logger.error(err)
        raise ValueError(err)
    url = dataset.get("url")
    existing_data_path = dataset.get("data_path", None)
    existing_files = dataset.get("files", {})

    data_path = set_data_path(data_name, data_path, existing_data_path)

    files_in_dir = get_file_list(data_path)

    # if existing_files is same as files_in_dir, then don't download/extract data
    if existing_files == files_in_dir:
        logger.info(f"{data_name} data is already present.")
    else:
        _ = download_data(data_name, data_path, url)
        extracted = extract_data(data_name, data_path, subdirs)
        if rearrange_files and extracted:
            rearrange_data(data_path, subdirs)

        # Update datasets JSON
        dataset["files"] = get_file_list(data_path)
        dataset["data_path"] = data_path

        save_config(
            {data_name: dataset},
            str(USER_JSON_PATH),
        )  # Save the updated dataset information

    # Return subdirectory paths if present
    if subdirs and not rearrange_files:
        return data_path, [os.path.join(data_path, subdir) for subdir in subdirs]
    else:
        return data_path

"""This module provides functions to fetch and load datasets."""
from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from sed.core.config import parse_config
from sed.core.config import save_config
from sed.core.logging import setup_logging
from sed.core.user_dirs import construct_module_dirs

# Configure logging
logger = setup_logging(__name__)

NAME = "datasets"
user_paths = construct_module_dirs(NAME)
json_path_user = user_paths["config"].joinpath(NAME).with_suffix(".json")
json_path_module = Path(os.path.dirname(__file__)).joinpath(NAME).with_suffix(".json")


def load_datasets_dict() -> dict:
    """
    Loads the datasets configuration dictionary from the user's datasets JSON file.

    If the file does not exist, it copies the default datasets JSON file from the module
    directory to the user's datasets directory.

    Returns:
        dict: The datasets dict loaded from user's datasets JSON file.
    """
    # check if datasets.json exists in user_config_dir
    if not os.path.exists(json_path_user):
        shutil.copy(json_path_module, json_path_user)
    datasets = parse_config(
        folder_config={},
        system_config=str(json_path_user),
        default_config=str(json_path_module),
        verbose=False,
    )
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
        data_path = existing_data_path or str(user_paths["data"] / data_name)
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


def download_data(
    data_name: str,
    data_path: str,
    data_url: str,
    chunk_size: int = 1024 * 32,
) -> bool:
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

    if os.path.exists(zip_file_path):
        existing_file_size = os.path.getsize(zip_file_path)
    else:
        existing_file_size = 0

    headers = {"Range": f"bytes={existing_file_size}-"}
    response = requests.get(data_url, headers=headers, stream=True)
    total_length = int(response.headers.get("content-length", 0))
    total_size = existing_file_size + total_length

    if response.status_code == 416:  # Range not satisfiable, file is already fully downloaded
        logger.info(f"{data_name} data is already fully downloaded.")
        return True

    mode = "ab" if existing_file_size > 0 else "wb"
    with open(zip_file_path, mode) as f, tqdm(
        total=total_size,
        initial=existing_file_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                pbar.update(len(chunk))

    logger.info("Download complete.")
    return True


def extract_data(data_name: str, data_path: str) -> bool:
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

    extracted_files = set()
    total_files = 0

    # Check if any subdirectory already contains files
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        total_files = len(zip_ref.infolist())
        for file in zip_ref.infolist():
            extracted_file_path = os.path.join(data_path, file.filename)
            if (
                os.path.exists(extracted_file_path)
                and os.path.getsize(extracted_file_path) == file.file_size
            ):
                extracted_files.add(file.filename)

    if len(extracted_files) == total_files:
        logger.info(f"{data_name} data is already fully extracted.")
        return True

    logger.info(f"Extracting {data_name} data...")
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        with tqdm(total=total_files, unit="file") as pbar:
            for file in zip_ref.infolist():
                if file.filename in extracted_files:
                    pbar.update(1)
                    continue
                zip_ref.extract(file, data_path)
                pbar.update(1)
    logger.info("Extraction complete.")
    return True


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
            logger.info(f"Rearranging files in {subdir}.")

            # Count the total number of files to move
            total_files = sum(len(files) for _, _, files in os.walk(source_path))

            with tqdm(total=total_files, unit="file") as pbar:
                for root, _, files in os.walk(source_path):
                    for file in files:
                        shutil.move(os.path.join(root, file), data_path)
                        pbar.update(1)

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
        extracted = extract_data(data_name, data_path)
        if rearrange_files and extracted:
            rearrange_data(data_path, subdirs)

        # Update datasets JSON
        dataset["files"] = get_file_list(data_path)
        dataset["data_path"] = data_path

        save_config(
            {data_name: dataset},
            str(json_path_user),
        )  # Save the updated dataset information

    # Return subdirectory paths if present
    if subdirs and not rearrange_files:
        return data_path, [os.path.join(data_path, subdir) for subdir in subdirs]
    else:
        return data_path

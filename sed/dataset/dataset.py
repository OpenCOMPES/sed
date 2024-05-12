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
if not os.path.exists(os.path.join(user_dirs.user_config_dir, "datasets.json")):
    shutil.copy(
        os.path.join(os.path.dirname(__file__), "datasets.json"),
        os.path.join(user_dirs.user_config_dir, "datasets.json"),
    )
DATASETS = json.load(open(os.path.join(user_dirs.user_config_dir, "datasets.json")))


def available_datasets():
    """Returns a list of available datasets."""
    return list(DATASETS.keys())


def fetch(data_name: str, data_path: str = None):
    """
    Fetches the specified data and extracts it to the given data path.

    Args:
        data_name (str): Name of the data to fetch.
        data_path (str): Path where the data should be stored. Default is the current directory.
    """
    if data_name not in DATASETS:
        logger.error(
            f"Data '{data_name}' is not available for fetching.\
                Available datasets are: {available_datasets()}",
        )
        return

    dataset_info = DATASETS[data_name]
    data_url = dataset_info["url"]
    subdirs = dataset_info.get("subdirs", [])
    rearrange_files = dataset_info.get("rearrange_files", False)
    existing_data_path = dataset_info.get("data_path", "")
    processed = dataset_info.get("processed", False)

    ## tell user that data might already exists
    if existing_data_path != data_path and data_path is not None:
        logger.info(
            f'{data_name} data might already exists at "{existing_data_path}",\
            unless deleted manually.',
        )

    if data_path is None:
        data_path = existing_data_path or user_dirs.user_data_dir
        logger.info(f'Data path not provided. Using path: "{data_path}"')

    zip_file_path = os.path.join(data_path, f"{data_name}.zip")

    # Check if data already exists
    if not os.path.exists(zip_file_path):
        logger.info(f"Downloading {data_name} data...")
        response = requests.get(data_url)
        with open(zip_file_path, "wb") as f:
            f.write(response.content)
        logger.info("Download complete.")
        processed = False
    else:
        logger.info(f"{data_name} data is already downloaded.")

    # Set Extract data flag to true, if not already extracted
    extract = True
    for subdir in subdirs:
        if os.path.isdir(os.path.join(data_path, subdir)):
            extract = False

    if extract:
        processed = False
        logger.info(f"Extracting {data_name} data...")
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(data_path)
        logger.info("Extraction complete.")
    else:
        logger.info(f"{data_name} data is already extracted.")

    # Move files if specified
    if rearrange_files and not processed:
        for subdir in subdirs:
            source_path = os.path.join(data_path, subdir)
            if os.path.isdir(source_path):
                logger.info(f"Rearranging files...")
                for root, dirs, files in os.walk(source_path):
                    for file in files:
                        shutil.move(os.path.join(root, file), data_path)
                logger.info("File movement complete.")
            else:
                logger.error(f"Subdirectory {subdir} not found.")
        logger.info("Rearranging complete.")

    ## add processed flag to dataset_info
    processed = True
    dataset_info["processed"] = processed
    dataset_info["data_path"] = data_path
    ## update datasets.json
    with open(os.path.join(user_dirs.user_config_dir, "datasets.json"), "w") as f:
        json.dump(DATASETS, f, indent=4)

    # if rearrange_files is not true and subdirs are present, return the subdir path
    if subdirs and not rearrange_files:
        return [os.path.join(data_path, subdir) for subdir in subdirs]
    else:
        return data_path

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


def available_datasets():
    """Returns a list of available datasets."""
    return list(DATASETS.keys())


def get_file_list(directory, ignore_zip=True):
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


def download_data(data_name, data_path, data_url):
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


def extract_data(data_name, data_path, subdirs):
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


def rearrange_data(data_name, data_path, subdirs, rearrange_files):
    subdirs = DATASETS.get(data_name).get("subdirs", [])
    data_path = DATASETS.get(data_name).get("data_path")
    rearrange_files = DATASETS.get(data_name).get("rearrange_files", False)
    # Move files if specified
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


def load_dataset(data_name: str, data_path: str = None):
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

    dataset = DATASETS.get(data_name)
    subdirs = dataset.get("subdirs", [])
    url = dataset.get("url")
    rearrange_files = dataset.get("rearrange_files", False)
    existing_data_path = dataset.get("data_path", None)
    existing_files = dataset.get("files", {})

    ## tell user that data might already exists
    if existing_data_path and data_path and existing_data_path != data_path:
        logger.info(
            f'{data_name} data might already exists at "{existing_data_path}", '
            "unless deleted manually.",
        )

    if data_path is None:
        if existing_data_path:
            data_path = existing_data_path
        else:
            ## create a new dir in user_dirs.user_data_dir with data_name
            data_path = os.path.join(user_dirs.user_data_dir, "datasets", data_name)
        logger.info(f'Data path not provided. Using path: "{data_path}"')

        if not os.path.exists(data_path):
            os.makedirs(data_path)

    files_in_dir = get_file_list(data_path)
    # if existing_files is same as files_in_dir, then don't download or process data
    if existing_files == files_in_dir:
        logger.info(f"{data_name} data is already downloaded and extracted.")
    else:
        download_data(data_name, data_path, url)
        extract_data(data_name, data_path, subdirs)
        rearrange_data(data_name, data_path, subdirs, rearrange_files)

        # update user datasets.json
        dataset["files"] = get_file_list(data_path)
        dataset["data_path"] = data_path
        with open(os.path.join(json_path), "w") as f:
            json.dump(DATASETS, f, indent=4)

    # if rearrange_files is not true and subdirs are present, return the subdir path
    if subdirs and not rearrange_files:
        return data_path, [os.path.join(data_path, subdir) for subdir in subdirs]
    else:
        return data_path

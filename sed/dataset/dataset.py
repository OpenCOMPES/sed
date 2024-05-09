import json
import logging
import os
import shutil
import zipfile

import requests

# Configure logging
logging.basicConfig(filename="dataset_fetch.log", level=logging.INFO)

DATASETS = json.load(open(os.path.join(os.path.dirname(__file__), "datasets.json")))


def available_datasets():
    """Returns a list of available datasets."""
    return list(DATASETS.keys())


def fetch(data_name, data_path="."):
    """
    Fetches the specified data and extracts it to the given data path.

    Args:
        data_name (str): Name of the data to fetch.
        data_path (str): Path where the data should be stored. Default is the current directory.
    """
    if data_name not in DATASETS:
        logging.error(f"Data '{data_name}' is not available for fetching.")
        return

    dataset_info = DATASETS[data_name]
    data_url = dataset_info["url"]
    subdirs = dataset_info.get("subdirs", [])
    resort_files = dataset_info.get("resort_files", False)
    zip_file_path = os.path.join(data_path, f"{data_name}.zip")

    # Check if data already exists
    if not os.path.exists(zip_file_path):
        logging.info(f"Downloading {data_name} data...")
        response = requests.get(data_url)
        with open(zip_file_path, "wb") as f:
            f.write(response.content)
        logging.info("Download complete.")
        downloaded = True
    else:
        logging.info(f"{data_name} data is already downloaded.")
        downloaded = False

    # Extract data if not already extracted
    extract = True
    for subdir in subdirs:
        if os.path.isdir(os.path.join(data_path, subdir)):
            extract = False

    if extract:
        logging.info(f"Extracting {data_name} data...")
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(data_path)
        logging.info("Extraction complete.")
    else:
        logging.info(f"{data_name} data is already extracted.")

    # Move files if specified
    if resort_files and downloaded:
        for subdir in subdirs:
            source_path = os.path.join(data_path, subdir)
            if os.path.isdir(source_path):
                logging.info(f"Moving files from {subdir} to {data_path}...")
                for root, dirs, files in os.walk(source_path):
                    for file in files:
                        shutil.move(os.path.join(root, file), data_path)
                # shutil.rmtree(source_path)
                logging.info("File movement complete.")
            else:
                logging.error(f"Source directory {subdir} not found.")

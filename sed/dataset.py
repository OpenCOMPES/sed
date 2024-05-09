import os
import shutil
import zipfile

import requests


DATASETS = {
    "WSe2": {
        "url": "https://zenodo.org/record/6369728/files/WSe2.zip",
        "subdirs": ["Scan049_1", "energycal_2019_01_08"],
    },
    "Gd_W110": {
        "url": "https://zenodo.org/records/10658470/files/single_event_data.zip",
        "subdirs": ["analysis_data", "calibration_data"],
        "resort_files": True,
    },
    "test": {
        "url": "https://syncandshare.desy.de/index.php/s/58y44ncLFWpggTS/download",
        "subdirs": ["PR_393"],
        "resort_files": True,
    },
}


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
        print(f"Data '{data_name}' is not available for fetching.")
        return

    dataset_info = DATASETS[data_name]
    data_url = dataset_info["url"]
    subdirs = dataset_info.get("subdirs", [])
    resort_files = dataset_info.get("resort_files", False)
    zip_file_path = os.path.join(data_path, f"{data_name}.zip")

    # Check if data already exists
    if not os.path.exists(zip_file_path):
        print(f"Downloading {data_name} data...")
        response = requests.get(data_url)
        with open(zip_file_path, "wb") as f:
            f.write(response.content)
        print("Download complete.")
    else:
        print(f"{data_name} data is already downloaded.")

    # Extract data if not already extracted
    extract = True
    for subdir in subdirs:
        if os.path.isdir(os.path.join(data_path, subdir)):
            extract = False

    if extract:
        print(f"Extracting {data_name} data...")
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(data_path)
        print("Extraction complete.")
    else:
        print(f"{data_name} data is already extracted.")

    # Move files if specified
    if resort_files:
        for subdir in subdirs:
            source_path = os.path.join(data_path, subdir)
            if os.path.isdir(source_path):
                print(f"Moving files from {subdir} to {data_path}...")
                for root, dirs, files in os.walk(source_path):
                    for file in files:
                        shutil.move(os.path.join(root, file), data_path)
                shutil.rmtree(source_path)
                print("File movement complete.")
            else:
                print(f"Source directory {subdir} not found.")

"""This module provides a Dataset class to download and extract datasets from web.
These datasets are defined in a JSON file. The Dataset class implements these features
Easy API:
from sed.dataset import datasets
datasets.get("NAME")
"""
from __future__ import annotations

import os
import shutil
import zipfile
from datetime import datetime
from importlib.util import find_spec

import requests
from tqdm.auto import tqdm

from sed.core.config import load_config
from sed.core.config import parse_config
from sed.core.config import save_config
from sed.core.config import USER_CONFIG_PATH
from sed.core.logging import setup_logging

package_dir = os.path.dirname(find_spec("sed").origin)

# Configure logging
logger = setup_logging("dataset")


class DatasetsManager:
    """Class to manage adding and removing datasets from the JSON file."""

    NAME = "datasets"
    FILENAME = NAME + ".json"
    json_path = {}
    json_path["user"] = os.path.join(USER_CONFIG_PATH, FILENAME)
    json_path["module"] = os.path.join(package_dir, "config", FILENAME)
    json_path["folder"] = "./" + FILENAME

    @staticmethod
    def load_datasets_dict() -> dict:
        """
        Loads the datasets configuration dictionary from the user's datasets JSON file.

        If the file does not exist, it copies the default datasets JSON file from the module
        directory to the user's datasets directory.

        Returns:
            dict: The datasets dict loaded from the user's datasets JSON file.
        """
        if not os.path.exists(DatasetsManager.json_path["user"]):
            shutil.copy(DatasetsManager.json_path["module"], DatasetsManager.json_path["user"])

        return parse_config(
            folder_config=DatasetsManager.json_path["folder"],
            user_config=DatasetsManager.json_path["user"],
            system_config={},
            default_config=DatasetsManager.json_path["module"],
            verbose=False,
            verify_config=False,
        )

    @staticmethod
    def add(data_name: str, info: dict, levels: list = ["user"]):
        """
        Adds a new dataset to the datasets JSON file.

        Args:
            data_name (str): Name of the dataset.
            info (dict): Information about the dataset.
            levels (list): List of levels to add the dataset to. Default is ["user"].
        """
        for level in levels:
            path = DatasetsManager.json_path[level]
            json_dict = load_config(path) if os.path.exists(path) else {}
            # if data_name already exists, throw error
            if data_name in json_dict:
                error_message = f"Dataset {data_name} already exists in {level} datasets.json."
                logger.error(error_message)
                raise ValueError(error_message)
            json_dict[data_name] = info
            save_config(json_dict, path)
            logger.info(f"Added {data_name} dataset to {level} datasets.json")

    @staticmethod
    def remove(data_name: str, levels: list = ["user"]):
        """
        Removes a dataset from the datasets JSON file.

        Args:
            data_name (str): Name of the dataset.
            levels (list): List of levels to remove the dataset from. Default is ["user"].
        """
        for level in levels:
            path = DatasetsManager.json_path[level]
            if os.path.exists(path):
                json_dict = load_config(path)
                del json_dict[data_name]
                save_config(json_dict, path, overwrite=True)
                logger.info(f"Removed {data_name} dataset from {level} datasets.json")


class Dataset:
    def __init__(self):
        self._datasets: dict = DatasetsManager.load_datasets_dict()
        self._dir: str = None
        self._subdirs: list[str] = None
        self._data_name: str = None
        self._state: dict = None
        self.subdirs: list[str] = None
        self.dir: str = None

    @property
    def available(self) -> list:
        """
        Returns a list of available datasets.

        Returns:
            list: List of available datasets.
        """
        self._datasets = DatasetsManager.load_datasets_dict()
        # remove Test from available datasets
        return [dataset for dataset in self._datasets if dataset != "Test"]

    def _check_dataset_availability(self) -> dict:
        """
        Checks if the specified dataset is available in the predefined list of datasets.

        Returns:
            dict: The dataset information if available.

        Raises:
            ValueError: If the dataset is not found in the predefined list.
        """
        if self._data_name not in self._datasets:
            error_message = (
                f"Data '{self._data_name}' is not available for fetching.\n"
                f"Available datasets are: {self.available}"
            )
            logger.error(error_message)
            raise ValueError(error_message)
        return self._datasets.get(self._data_name)

    @property
    def data_name(self) -> str:
        """Get the data name."""
        return self._data_name

    @data_name.setter
    def data_name(self, value: str):
        """Set the data name and update the state."""
        self._data_name = value
        self._datasets = DatasetsManager.load_datasets_dict()
        self._state = self._check_dataset_availability()
        self._subdirs = self._state.get("subdirs", [])

    @property
    def existing_data_paths(self) -> list:
        """Get paths where dataset exists."""
        if not self._data_name:
            raise ValueError("Data name not set.")
        return self._state.get("data_path", [])

    def _set_data_dir(
        self,
        root_dir: str,
        use_existing: bool,
    ):
        """
        Determines and sets the data path for a dataset. If a data path is not provided,
        it uses the existing data path or creates a new one. It also notifies the user
        if the specified data path differs from an existing data path.

        Args:
            root_dir (str): The desired path where the dataset should be stored.
            use_existing (bool): Whether to use the existing data path.
        """
        existing_data_path = self.existing_data_paths[0] if self.existing_data_paths else None
        if use_existing and existing_data_path:
            if existing_data_path != root_dir:
                logger.info(
                    f"Not downloading {self._data_name} data as it already exists "
                    f'at "{existing_data_path}".\n'
                    "Set 'use_existing' to False if you want to download to a new location.",
                )
            dir_ = existing_data_path
            path_source = "existing"
        else:
            if not root_dir:
                root_dir = os.getcwd()
                path_source = "default"
            else:
                path_source = "specified"
            dir_ = os.path.join(root_dir, DatasetsManager.NAME, self._data_name)

        self._dir = os.path.abspath(dir_)
        logger.info(f'Using {path_source} data path for "{self._data_name}": "{self._dir}"')

        if not os.path.exists(self._dir):
            os.makedirs(self._dir)
            logger.info(f"Created new directory at {self._dir}")

    def _get_file_list(self, ignore_zip: bool = True) -> list[str]:
        """
        Returns a list of file paths in the directory and its subdirectories.

        Args:
            ignore_zip (bool): Whether to ignore ZIP files. Default is True.

        Returns:
            List[str]: List of file paths in the directory and its subdirectories.
        """
        file_paths = []

        # Walk through directory and its subdirectories
        for root, _, files in os.walk(self._dir):
            for file in files:
                # Ignore hidden files and ZIP files if ignore_zip is True
                if not file.startswith(".") and (not ignore_zip or not file.endswith(".zip")):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self._dir)
                    file_paths.append(relative_path)

        return file_paths

    def _download_data(
        self,
        data_url: str,
        chunk_size: int = 1024 * 32,
    ):
        """
        Downloads data from the specified URL.

        Args:
            data_url (str): URL of the data.
            chunk_size (int): Size of the chunks to download. Default is 32 KB.
        """
        zip_file_path = os.path.join(self._dir, f"{self._data_name}.zip")

        if os.path.exists(zip_file_path):
            existing_file_size = os.path.getsize(zip_file_path)
        else:
            existing_file_size = 0

        headers = {"Range": f"bytes={existing_file_size}-"}
        response = requests.get(data_url, headers=headers, stream=True)

        if response.status_code == 416:  # Range not satisfiable, file is already fully downloaded
            logger.info(f"{self._data_name} data is already fully downloaded.")

        total_length = int(response.headers.get("content-length", 0))
        if response.status_code == 206:  # Partial content
            total_size = existing_file_size + total_length
        else:
            total_size = total_length if total_length > 0 else None

        mode = "ab" if existing_file_size > 0 else "wb"
        with open(zip_file_path, mode) as f, tqdm(
            total=total_size,
            initial=existing_file_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            disable=total_size is None,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    pbar.update(len(chunk))

        logger.info("Download complete.")

    def _extract_data(self, remove_zip: bool = True) -> None:
        """
        Extracts data from a ZIP file.
        """
        zip_file_path = os.path.join(self._dir, f"{self._data_name}.zip")

        extracted_files = set()
        total_files = 0

        # Check if any subdirectory already contains files
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            total_files = len(zip_ref.infolist())
            for file in zip_ref.infolist():
                extracted_file_path = os.path.join(self._dir, file.filename)
                if (
                    os.path.exists(extracted_file_path)
                    and os.path.getsize(extracted_file_path) == file.file_size
                ):
                    extracted_files.add(file.filename)

        if len(extracted_files) == total_files:
            logger.info(f"{self._data_name} data is already fully extracted.")
            return

        logger.info(f"Extracting {self._data_name} data...")
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            with tqdm(total=total_files, unit="file") as pbar:
                for file in zip_ref.infolist():
                    if file.filename in extracted_files:
                        pbar.update(1)
                        continue
                    zip_ref.extract(file, self._dir)
                    # Preserve original timestamp
                    timestamp = datetime(*file.date_time).timestamp()
                    os.utime(os.path.join(self._dir, file.filename), (timestamp, timestamp))
                    pbar.update(1)
        logger.info(f"{self._data_name} data extracted successfully.")

        if remove_zip:
            os.remove(zip_file_path)
            logger.info(f"Removed {self._data_name}.zip file.")

    def _rearrange_data(self) -> None:
        """
        Moves files to the main directory if specified.
        """
        for subdir in self._subdirs:
            source_path = os.path.join(self._dir, subdir)
            if os.path.isdir(source_path):
                logger.info(f"Rearranging files in {subdir}.")

                # Count the total number of files to move
                total_files = sum(len(files) for _, _, files in os.walk(source_path))

                with tqdm(total=total_files, unit="file") as pbar:
                    for root, _, files in os.walk(source_path):
                        for file in files:
                            shutil.move(os.path.join(root, file), os.path.join(self._dir, file))
                            pbar.update(1)

                logger.info("File movement complete.")
                shutil.rmtree(source_path)
            else:
                error_message = f"Subdirectory {subdir} not found."
                logger.error(error_message)
                raise FileNotFoundError(error_message)
        self._subdirs = []
        logger.info("Rearranging complete.")

    def get(
        self,
        data_name: str,
        **kwargs,
    ):
        """
        Fetches the specified data and extracts it to the given data path.

        Args:
            data_name (str): Name of the data to fetch.
            root_dir (str): Path where the data should be stored. Default is the current directory.
            use_existing (bool): Whether to use the existing data path. Default is True.
            remove_zip (bool): Whether to remove the ZIP file after extraction. Default is True.
            ignore_zip (bool): Whether to ignore ZIP files when listing files. Default is True.
        """
        self.data_name = data_name  # sets self._data_name and self._state

        self._set_data_dir(
            kwargs.get("root_dir", None),
            kwargs.get("use_existing", True),
        )
        files_in_dir = self._get_file_list(kwargs.get("ignore_zip", True))
        file_list: list = self._state.get("files", [])

        # if all files are present, skip download
        if file_list and all(file in files_in_dir for file in file_list):
            logger.info(f"{self._data_name} data is already present.")
        else:
            url: str = self._state.get("url")
            self._download_data(url)
            self._extract_data(kwargs.get("remove_zip", True))
            if self._state.get("rearrange_files", False):
                self._rearrange_data()

            # Update datasets JSON
            self._state["files"] = self._get_file_list()
            existing = set(self.existing_data_paths)
            existing.add(self._dir)
            self._state["data_path"] = list(existing)
            save_config({self._data_name: self._state}, DatasetsManager.json_path["user"])

        self.subdirs = [os.path.join(self._dir, subdir) for subdir in self._subdirs]
        self.dir = self._dir

    def remove(self, data_name: str, instance: str = "all"):
        """
        Removes directories of all or defined instances of the specified dataset.

        Args:
            data_name (str): Name of the dataset.
            instance (str): Name of the instance to remove. Default is "all".
        """
        self.data_name = data_name
        existing = self.existing_data_paths
        if not existing:
            logger.info(f"{data_name} data is not present.")
            return

        if instance == "all":
            for path in existing:
                if os.path.exists(path):
                    shutil.rmtree(path)
                    logger.info(f"Removed {path}")
            existing.clear()
        else:
            if instance not in existing:
                logger.info(f"{instance} instance of {data_name} data is not present.")
                return
            if os.path.exists(instance):
                shutil.rmtree(instance)
                logger.info(f"Removed {instance}")
            existing.remove(instance)

        # Update datasets JSON
        self._state["data_path"] = existing
        save_config({self._data_name: self._state}, DatasetsManager.json_path["user"])


dataset = Dataset()

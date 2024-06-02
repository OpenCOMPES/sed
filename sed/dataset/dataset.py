"""This module provides functions to fetch and load datasets."""
from __future__ import annotations

import os
import shutil
import zipfile

import requests
from tqdm import tqdm

from sed.core.config import parse_config
from sed.core.config import save_config
from sed.core.logging import setup_logging
from sed.core.user_dirs import USER_CONFIG_PATH

# Configure logging
logger = setup_logging(__name__)


class Dataset:
    NAME = "datasets"
    json_path_user = os.path.join(USER_CONFIG_PATH, NAME + ".json")
    json_path_module = os.path.join(os.path.dirname(__file__), NAME + ".json")

    def __init__(self):
        self.datasets = self._load_datasets_dict()
        self.dir: str = None
        self.subdirs: list[str] = None

    def _load_datasets_dict(self) -> dict:
        """
        Loads the datasets configuration dictionary from the user's datasets JSON file.

        If the file does not exist, it copies the default datasets JSON file from the module
        directory to the user's datasets directory.

        Returns:
            dict: The datasets dict loaded from user's datasets JSON file.
        """
        # check if datasets.json exists in user_config_dir
        if not os.path.exists(self.json_path_user):
            shutil.copy(self.json_path_module, self.json_path_user)
        datasets = parse_config(
            folder_config={},
            system_config=str(self.json_path_user),
            default_config=str(self.json_path_module),
            verbose=False,
        )
        return datasets

    @property
    def available(self) -> list:
        """
        Returns a list of available datasets.

        Returns:
            list: List of available datasets.
        """
        return list(self.datasets)

    def _check_dataset_availability(self) -> dict:
        """
        Checks if the specified dataset is available in the predefined list of datasets.

        Args:
            data_name (str): The name of the dataset to check.

        Returns:
            dict: The dataset information if available.

        Raises:
            ValueError: If the dataset is not found in the predefined list.
        """
        if self._data_name not in self.datasets:
            error_message = f"Data '{self._data_name}' is not available for fetching.\
                Available datasets are: {self.available}"
            logger.error(error_message)
            raise ValueError(error_message)
        return self.datasets.get(self._data_name)

    def _set_data_dir(
        self,
        data_name: str,
        root_dir: str,
        existing_data_path: str,
        use_existing: bool,
    ):
        """
        Determines and sets the data path for a dataset. If a data path is not provided,
        it uses the existing data path or creates a new one. It also notifies the user
        if the specified data path differs from an existing data path.

        Args:
            data_name (str): The name of the dataset.
            root_dir (str, optional): The desired path where the dataset should be stored.
                                        Defaults to the current directory.
            existing_data_path (str, optional): The path where the dataset currently exists.
            use_existing (bool): Whether to use the existing data path.
        """
        if use_existing:
            # Notify the user if data might already exist
            if existing_data_path and root_dir and existing_data_path != root_dir:
                logger.warning(
                    f'Using {data_name} data that already exists at "{existing_data_path}", '
                    "Set 'use_existing' to False if you want to download to a new location.",
                )

        if not use_existing:
            path_source = "existing"
            dir_ = os.path.abspath(existing_data_path)
        else:
            path_source = "specified"
            if not root_dir:
                path_source = "default"
                root_dir = os.getcwd()
            dir_ = existing_data_path or os.path.join(root_dir, self.NAME, data_name)
        self.dir = os.path.abspath(dir_)  # absolute path
        logger.info(f'Using {path_source} data path for "{data_name}": "{self.dir}"')

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
            logger.info(f"Created new directory at {self.dir}")

    def _get_file_list(self, ignore_zip: bool = True) -> dict:
        """
        Returns a dictionary containing lists of files in each subdirectory
        and files in the main directory.

        Args:
            ignore_zip (bool): Whether to ignore ZIP files. Default is True.

        Returns:
            dict: Dictionary containing lists of files in each subdirectory and files
                    in the main directory.
        """
        result = {}
        main_dir_files = []

        # List all files and directories in the given directory
        all_files_and_dirs = os.listdir(self.dir)

        # Filter out hidden files and directories
        visible_files_and_dirs = [item for item in all_files_and_dirs if not item.startswith(".")]

        # Organize files into dictionary structure
        for item in visible_files_and_dirs:
            item_path = os.path.join(self.dir, item)
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

    def _download_data(
        self,
        data_url: str,
        chunk_size: int = 1024 * 32,
    ) -> bool:
        """
        Downloads data from the specified URL.

        Args:
            data_url (str): URL of the data.

        Returns:
            bool: True if the data was downloaded successfully,
                    False if the data is already downloaded.
        """
        zip_file_path = os.path.join(self.dir, f"{self._data_name}.zip")

        if os.path.exists(zip_file_path):
            existing_file_size = os.path.getsize(zip_file_path)
        else:
            existing_file_size = 0

        headers = {"Range": f"bytes={existing_file_size}-"}
        response = requests.get(data_url, headers=headers, stream=True)
        total_length = int(response.headers.get("content-length", 0))
        total_size = existing_file_size + total_length

        if response.status_code == 416:  # Range not satisfiable, file is already fully downloaded
            logger.info(f"{self._data_name} data is already fully downloaded.")
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

    def _extract_data(self):
        """
        Extracts data from a ZIP file.
        """
        zip_file_path = os.path.join(self.dir, f"{self._data_name}.zip")

        extracted_files = set()
        total_files = 0

        # Check if any subdirectory already contains files
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            total_files = len(zip_ref.infolist())
            for file in zip_ref.infolist():
                extracted_file_path = os.path.join(self.dir, file.filename)
                if (
                    os.path.exists(extracted_file_path)
                    and os.path.getsize(extracted_file_path) == file.file_size
                ):
                    extracted_files.add(file.filename)

        if len(extracted_files) == total_files:
            logger.info(f"{self._data_name} data is already fully extracted.")
            return True

        logger.info(f"Extracting {self._data_name} data...")
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            with tqdm(total=total_files, unit="file") as pbar:
                for file in zip_ref.infolist():
                    if file.filename in extracted_files:
                        pbar.update(1)
                        continue
                    zip_ref.extract(file, self.dir)
                    pbar.update(1)
        logger.info("Extraction complete.")
        return True

    def _rearrange_data(self) -> None:
        """
        Moves files to the main directory if specified.
        """
        for subdir in self.subdirs:
            source_path = os.path.join(self.dir, subdir)
            if os.path.isdir(source_path):
                logger.info(f"Rearranging files in {subdir}.")

                # Count the total number of files to move
                total_files = sum(len(files) for _, _, files in os.walk(source_path))

                with tqdm(total=total_files, unit="file") as pbar:
                    for root, _, files in os.walk(source_path):
                        for file in files:
                            shutil.move(os.path.join(root, file), self.dir)
                            pbar.update(1)

                logger.info("File movement complete.")
                shutil.rmtree(source_path)
            else:
                error_message = f"Subdirectory {subdir} not found."
                logger.error(error_message)
                raise FileNotFoundError(error_message)

        logger.info("Rearranging complete.")

    def get(
        self,
        data_name: str,
        root_dir: str = None,
        use_existing: bool = True,
    ):
        """
        Fetches the specified data and extracts it to the given data path.

        Args:
            data_name (str): Name of the data to fetch.
            root_dir (str): Path where the data should be stored. Default is the current directory.

        Returns:
            str | tuple[str, list[str]]: Path to the dataset or a tuple containing the path to the
            dataset and subdirectories.
        """
        self._data_name = data_name
        self._state = self._check_dataset_availability()
        subdirs = self._state.get("subdirs", [])
        rearrange_files = self._state.get("rearrange_files", False)
        if rearrange_files and not self.subdirs:
            err = (
                f"Rearrange_files is set to True but no subdirectories are defined for {data_name}."
            )
            logger.error(err)
            raise ValueError(err)
        url = self._state.get("url")
        existing_data_path = self._state.get("data_path", None)
        existing_files = self._state.get("files", {})

        self.dir = self._set_data_dir(data_name, root_dir, existing_data_path, use_existing)

        files_in_dir = self._get_file_list()

        # if existing_files is same as files_in_dir, then don't download/extract data
        if existing_files == files_in_dir:
            logger.info(f"{data_name} data is already present.")
        else:
            _ = self._download_data(url)
            extracted = self._extract_data()
            if rearrange_files and extracted:
                self._rearrange_data()

            # Update datasets JSON
            self._state["files"] = self._get_file_list()
            self._state["data_path"] = self.dir

            save_config(
                {data_name: self._state},
                str(self.json_path_user),
            )  # Save the updated dataset information

        # Return subdirectory paths if present
        if subdirs and not rearrange_files:
            self.subdirs = [os.path.join(self.dir, subdir) for subdir in subdirs]


datasets = Dataset()

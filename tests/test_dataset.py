"""This code performs several tests for the dataset module.

The tests cover the following functionalities:
- Checking available datasets
- Checking dataset availability
- Setting root directory for datasets
- Getting file list from dataset directory
- Downloading dataset
- Extracting dataset
- Rearranging dataset
- Adding and removing datasets using DatasetsManager
"""
from __future__ import annotations

import io
import json
import os
import zipfile
from importlib.util import find_spec

import pytest

from sed.core.config import USER_CONFIG_PATH
from sed.dataset import dataset as ds
from sed.dataset import DatasetsManager as dm

package_dir = os.path.dirname(find_spec("sed").origin)
json_path = os.path.join(package_dir, "config/datasets.json")


@pytest.fixture
def zip_buffer():
    """Fixture to create an in-memory zip file buffer with test files."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("test_file.txt", "This is a test file inside the zip.")
        # add a subdir
        zip_file.writestr("subdir/test_subdir.txt", "This is a test file inside the subdir.")
    return zip_buffer


@pytest.fixture
def zip_filepath(fs, zip_buffer, tmp_path):
    """Fixture to create a temporary directory and write the zip buffer to a file."""
    test_dir = tmp_path / "datasets" / "Test"
    fs.create_dir(test_dir)
    with open(test_dir / "Test.zip", "wb") as f:
        f.write(zip_buffer.getvalue())
    return test_dir


def test_available_datasets():
    """Checks the available datasets by comparing with the loaded datasets dictionary."""
    all_dsets = dm.load_datasets_dict()
    del all_dsets["Test"]
    assert ds.available == list(all_dsets.keys())


def test_check_dataset_availability():
    """Checks that all available datasets are loaded and tests if an error is raised for
    unknown datasets."""
    datasets = dm.load_datasets_dict()
    # return dataset information if available
    for data_name in datasets.keys():
        ds._data_name = data_name
        assert datasets[data_name] == ds._check_dataset_availability()

    # raise error if dataset not found
    with pytest.raises(ValueError):
        ds._data_name = "UnknownDataset"
        ds._check_dataset_availability()


def test_set_root_dir(tmp_path):
    """Checks if all cases of setting root datasets directory."""
    # test with existing data path
    ds.data_name = "Test"
    ds._state["data_path"] = [str(tmp_path / "test" / "data")]
    ds._set_data_dir(root_dir=str(tmp_path / "test" / "data"), use_existing=True)
    assert ds._dir == str((tmp_path / "test" / "data").resolve())

    # test without existing data path
    ds._state["data_path"] = []
    ds._set_data_dir(root_dir=str(tmp_path / "test" / "data"), use_existing=True)
    assert ds._dir == str((tmp_path / "test" / "data" / "datasets" / "Test").resolve())

    # test without data path and existing data path
    ds._set_data_dir(root_dir=None, use_existing=True)
    assert f"{os.getcwd()}/datasets/Test" == ds._dir

    # test with provided data path different from existing data path
    ds._state["data_path"] = [str(tmp_path / "test" / "data1")]
    ds._set_data_dir(root_dir=str(tmp_path / "test" / "data"), use_existing=True)
    assert ds._dir == str((tmp_path / "test" / "data1").resolve())
    ds._set_data_dir(root_dir=str(tmp_path / "test" / "data"), use_existing=False)
    assert ds._dir == str((tmp_path / "test" / "data" / "datasets" / "Test").resolve())


def test_get_file_list(fs, tmp_path):
    """Test to get the list of files in the dataset directory, including and excluding zip files."""
    test_dir = tmp_path / "test" / "data"
    fs.create_file(test_dir / "file.txt")
    fs.create_file(test_dir / "subdir" / "file.txt")
    fs.create_file(test_dir / "subdir" / "file.zip")
    fs.create_file(test_dir / "file.zip")
    ds._dir = str(test_dir)
    assert ["file.txt", "subdir/file.txt"] == ds._get_file_list()

    assert ["file.txt", "file.zip", "subdir/file.txt", "subdir/file.zip"] == ds._get_file_list(
        ignore_zip=False,
    )


def test_download_data(fs, requests_mock, zip_buffer, tmp_path):
    """Test to download a dataset from a URL and verify the downloaded zip file."""
    test_dir = tmp_path / "test"
    fs.create_dir(test_dir)
    data_url = "http://test.com/files/file.zip"
    requests_mock.get(data_url, content=zip_buffer.getvalue())
    ds._data_name = "Test"
    ds._state = {"data_path": []}
    ds._set_data_dir(root_dir=str(test_dir), use_existing=True)
    ds._download_data(data_url)
    assert os.path.exists(test_dir / "datasets" / "Test" / "Test.zip")


def test_extract_data(zip_filepath):
    """Test to extract files from the dataset zip file and verify the extracted files."""
    ds._data_name = "Test"
    ds._dir = str(zip_filepath)
    ds._extract_data()
    assert os.path.exists(zip_filepath / "test_file.txt")
    assert os.path.exists(zip_filepath / "subdir" / "test_subdir.txt")


def test_rearrange_data(zip_filepath):
    """Test to rearrange files in the dataset directory and verify the rearranged files."""
    ds._data_name = "Test"
    ds._dir = str(zip_filepath)
    ds._subdirs = ["subdir"]
    ds._extract_data()
    ds._rearrange_data()
    assert os.path.exists(zip_filepath / "test_file.txt")
    assert os.path.exists(zip_filepath / "test_subdir.txt")
    assert not os.path.exists(zip_filepath / "subdir")

    with pytest.raises(FileNotFoundError):
        ds._subdirs = ["non_existing_subdir"]
        ds._rearrange_data()


def test_get_remove_dataset(requests_mock, zip_buffer, tmp_path):
    """Test to get a dataset, verify its directory and files, and then remove the dataset."""
    json_path_user = tmp_path / USER_CONFIG_PATH / "datasets.json"
    data_name = "Test"
    _ = dm.load_datasets_dict()  # to ensure datasets.json is in user dir

    ds.remove(data_name)

    data_url = "http://test.com/files/file.zip"
    requests_mock.get(data_url, content=zip_buffer.getvalue())

    ds.get(data_name, root_dir=str(tmp_path))
    assert ds.dir == str((tmp_path / "datasets" / data_name).resolve())

    # check if subdir is removed after rearranging
    assert not os.path.exists(tmp_path / "datasets" / "Test" / "subdir")

    # check datasets file to now have data_path listed
    datasets_json = json.load(open(json_path_user))
    assert datasets_json[data_name]["data_path"]
    assert datasets_json[data_name]["files"]
    ds.remove(data_name)

    assert not os.path.exists(tmp_path / "datasets" / data_name)

    ds.get(data_name, root_dir=str(tmp_path))
    ds.get(data_name, root_dir=str(tmp_path))
    ds.remove(data_name, ds.existing_data_paths[0])


def test_datasets_manager(tmp_path):  # noqa: ARG001
    """Tests adds a dataset using DatasetsManager, verifies its addition,
    removes it and checks for error raised."""
    dm.add(
        "Test_DM",
        {"url": "http://test.com/files/file.zip", "subdirs": ["subdir"]},
        levels=["folder", "user"],
    )
    datasets_json = json.load(open(dm.json_path["folder"]))
    assert datasets_json["Test_DM"]
    assert datasets_json["Test_DM"]["url"] == "http://test.com/files/file.zip"
    assert datasets_json["Test_DM"]["subdirs"] == ["subdir"]

    dm.remove("Test_DM", levels=["folder"])
    datasets_json = json.load(open(dm.json_path["folder"]))
    with pytest.raises(KeyError):
        datasets_json["Test_DM"]

    datasets_json = json.load(open(dm.json_path["user"]))
    assert datasets_json["Test_DM"]

    dm.remove("Test_DM", levels=["user"])
    datasets_json = json.load(open(dm.json_path["user"]))
    with pytest.raises(KeyError):
        datasets_json["Test_DM"]

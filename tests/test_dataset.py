"""This code  performs several tests for the dataset module.
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
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("test_file.txt", "This is a test file inside the zip.")
        # add a subdir
        zip_file.writestr("subdir/test_subdir.txt", "This is a test file inside the subdir.")
    return zip_buffer


@pytest.fixture
def zip_file(fs, zip_buffer):
    fs.create_dir("test/datasets/Test")
    with open("test/datasets/Test/Test.zip", "wb") as f:
        f.write(zip_buffer.getvalue())


def test_available_datasets():
    all_dsets = dm.load_datasets_dict()
    del all_dsets["Test"]
    assert ds.available == list(all_dsets.keys())


def test_check_dataset_availability():
    datasets = dm.load_datasets_dict()
    # return dataset information if available
    for data_name in datasets.keys():
        ds._data_name = data_name
        assert datasets[data_name] == ds._check_dataset_availability()

    # raise error if dataset not found
    with pytest.raises(ValueError):
        ds._data_name = "UnknownDataset"
        ds._check_dataset_availability()


def test_set_root_dir():
    # test with existing data path
    ds.data_name = "Test"
    ds._state["data_path"] = ["test/data"]
    ds._set_data_dir(root_dir="test/data", use_existing=True)
    assert os.path.abspath("test/data/") == ds._dir

    # test without existing data path
    ds._state["data_path"] = []
    ds._set_data_dir(root_dir="test/data", use_existing=True)
    assert os.path.abspath("test/data/datasets/Test") == ds._dir

    # test without data path and existing data path
    ds._set_data_dir(root_dir=None, use_existing=True)
    assert os.path.abspath("./datasets/Test") == ds._dir

    # test with provided data path different from existing data path
    ds._state["data_path"] = ["test/data1"]
    ds._set_data_dir(root_dir="test/data", use_existing=True)
    assert os.path.abspath("test/data1/") == ds._dir
    ds._set_data_dir(root_dir="test/data", use_existing=False)
    assert os.path.abspath("test/data/datasets/Test") == ds._dir


def test_get_file_list(fs):
    fs.create_file("test/data/file.txt")
    fs.create_file("test/data/subdir/file.txt")
    fs.create_file("test/data/subdir/file.zip")
    fs.create_file("test/data/file.zip")
    ds._dir = "test/data"
    assert ["file.txt", "subdir/file.txt"] == ds._get_file_list()

    assert ["file.txt", "file.zip", "subdir/file.txt", "subdir/file.zip"] == ds._get_file_list(
        ignore_zip=False,
    )


def test_download_data(fs, requests_mock, zip_buffer):
    fs.create_dir("test")
    data_url = "http://test.com/files/file.zip"
    requests_mock.get(data_url, content=zip_buffer.getvalue())
    ds._data_name = "Test"
    ds._state = {"data_path": []}
    ds._set_data_dir(root_dir="test", use_existing=True)
    ds._download_data(data_url)
    assert os.path.exists("test/datasets/Test/Test.zip")

    # assert not ds._download_data("data", "test/data/", data_url)  # already exists


def test_extract_data(zip_file):  # noqa: ARG001
    ds._data_name = "Test"
    ds._dir = "test/datasets/Test/"
    ds._extract_data()
    assert os.path.exists("test/datasets/Test/test_file.txt")
    assert os.path.exists("test/datasets/Test/subdir/test_subdir.txt")


def test_rearrange_data(zip_file):  # noqa: ARG001
    ds._data_name = "Test"
    ds._dir = "test/datasets/Test/"
    ds._subdirs = ["subdir"]
    ds._extract_data()
    ds._rearrange_data()
    assert os.path.exists("test/datasets/Test/test_file.txt")
    assert os.path.exists("test/datasets/Test/test_subdir.txt")
    assert not os.path.exists("test/datasets/Test/subdir")

    with pytest.raises(FileNotFoundError):
        ds._subdirs = ["non_existing_subdir"]
        ds._rearrange_data()


def test_get_remove_dataset(requests_mock, zip_buffer):
    json_path_user = USER_CONFIG_PATH.joinpath("datasets.json")
    data_name = "Test"
    _ = dm.load_datasets_dict()  # to ensure datasets.json is in user dir

    ds.remove(data_name)

    data_url = "http://test.com/files/file.zip"
    requests_mock.get(data_url, content=zip_buffer.getvalue())

    ds.get(data_name)
    assert ds.dir == os.path.abspath(os.path.join("./datasets", data_name))

    # check if subdir is removed after rearranging
    assert not os.path.exists("./datasets/Test/subdir")

    # check datasets file to now have data_path listed
    datasets_json = json.load(open(json_path_user))
    assert datasets_json[data_name]["data_path"]
    assert datasets_json[data_name]["files"]
    ds.remove(data_name)

    assert not os.path.exists(os.path.join("./datasets", data_name))

    ds.get(data_name)
    ds.get(data_name)
    ds.remove(data_name, ds.existing_data_paths[0])


def test_datasets_manager():
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

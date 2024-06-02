import io
import json
import os
import shutil
import zipfile
from importlib.util import find_spec

import pytest

import sed.dataset as ds
from sed.core.user_dirs import USER_CONFIG_PATH
from sed.core.user_dirs import USER_DATA_PATH
from sed.core.user_dirs import USER_LOG_PATH

package_dir = os.path.dirname(find_spec("sed").origin)
json_path = os.path.join(package_dir, "dataset/datasets.json")


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
    fs.create_dir("test/data")
    with open("test/data/data.zip", "wb") as f:
        f.write(zip_buffer.getvalue())


def test_available_datasets():
    assert ds.available_datasets() == list(ds.load_datasets_dict().keys())


def test_check_dataset_availability():
    datasets = ds.load_datasets_dict()
    # return dataset information if available
    for data_name in datasets.keys():
        assert datasets[data_name] == ds.check_dataset_availability(data_name)

    # raise error if dataset not found
    with pytest.raises(ValueError):
        ds.check_dataset_availability("UnknownDataset")


def test_set_data_path():
    data_name = "Test"
    # test with existing data path
    data_path = "test/data"
    existing_data_path = "test/data"
    assert data_path == ds.set_data_path(data_name, data_path, existing_data_path)

    # test without existing data path
    data_path = "test/data"
    existing_data_path = None
    assert data_path == ds.set_data_path(data_name, data_path, existing_data_path)

    # test without data path and existing data path
    data_path = None
    existing_data_path = None
    assert str(USER_DATA_PATH.joinpath("datasets", data_name)) == ds.set_data_path(
        data_name,
        data_path,
        existing_data_path,
    )

    # test with provided data path different from existing data path
    data_path = "test/data"
    existing_data_path = "test/data1"
    assert data_path == ds.set_data_path(data_name, data_path, existing_data_path)
    # check in user log path for the warning
    log_file = USER_LOG_PATH.joinpath("sed.log")
    with open(log_file) as f:
        assert f'{data_name} data might already exists at "{existing_data_path}"' in f.read()


def test_get_file_list(fs):
    fs.create_file("test/data/file.txt")
    fs.create_file("test/data/subdir/file.txt")
    fs.create_file("test/data/subdir/file.zip")
    fs.create_file("test/data/file.zip")

    assert {"main_directory": ["file.txt"], "subdir": ["file.txt"]} == ds.get_file_list("test/data")

    assert {
        "main_directory": ["file.txt", "file.zip"],
        "subdir": ["file.txt", "file.zip"],
    } == ds.get_file_list("test/data", ignore_zip=False)


def test_download_data(fs, requests_mock, zip_buffer):
    fs.create_dir("test/data")
    data_url = "http://test.com/files/file.zip"
    requests_mock.get(data_url, content=zip_buffer.getvalue())
    assert ds.download_data("data", "test/data/", data_url)
    assert os.path.exists("test/data/data.zip")

    # assert not ds.download_data("data", "test/data/", data_url)  # already exists


def test_extract_data(zip_file):  # noqa: ARG001
    assert ds.extract_data("data", "test/data")
    assert os.path.exists("test/data/test_file.txt")
    assert os.path.exists("test/data/subdir/test_subdir.txt")


def test_rearrange_data(zip_file):  # noqa: ARG001
    ds.extract_data("data", "test/data")
    assert os.path.exists("test/data/subdir")
    ds.rearrange_data("test/data", ["subdir"])
    assert os.path.exists("test/data/test_file.txt")
    assert os.path.exists("test/data/test_subdir.txt")
    assert ~os.path.exists("test/data/subdir")

    with pytest.raises(FileNotFoundError):
        ds.rearrange_data("test/data", ["non_existing_subdir"])


def test_load_dataset(requests_mock, zip_buffer):
    json_path_user = USER_CONFIG_PATH.joinpath("datasets", "datasets.json")
    data_name = "Test"
    _ = ds.load_datasets_dict()  # to ensure datasets.json is in user dir
    data_path_user = USER_DATA_PATH.joinpath("datasets", data_name)

    if os.path.exists(data_path_user):
        # remove dir even if it is not empty
        shutil.rmtree(data_path_user)

    data_url = "http://test.com/files/file.zip"
    requests_mock.get(data_url, content=zip_buffer.getvalue())

    paths = ds.load_dataset(data_name)
    assert paths == str(USER_DATA_PATH.joinpath("datasets", data_name))

    # check if subdir is removed after rearranging
    assert not os.path.exists(str(USER_DATA_PATH.joinpath("datasets", "Test", "subdir")))

    # check datasets file to now have data_path listed
    json_path_user = USER_CONFIG_PATH.joinpath("datasets", "datasets.json")
    datasets_json = json.load(open(json_path_user))
    assert datasets_json[data_name]["data_path"]
    assert datasets_json[data_name]["files"]

    paths = ds.load_dataset(data_name)

    shutil.rmtree(data_path_user)

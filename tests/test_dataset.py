"""This code performs several tests for the dataset module.
"""
from __future__ import annotations

import io
import json
import os
import zipfile
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import pytest

from sed.dataset import dataset as ds
from sed.dataset import DatasetsManager as dm

package_dir = os.path.dirname(find_spec("sed").origin)
json_path = os.path.join(package_dir, "config/datasets.json")

# cspell:ignore pytestmark xdist
pytestmark = pytest.mark.xdist_group(name="dataset_tests")


@pytest.fixture
def zip_buffer():
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("test_file.txt", "This is a test file inside the zip.")
        zip_file.writestr("subdir/test_subdir.txt", "This is a test file inside the subdir.")
    return zip_buffer


@pytest.fixture
def zip_file(tmp_path, zip_buffer):
    test_dir = tmp_path / "datasets" / "Test"
    test_dir.mkdir(parents=True)
    zip_path = test_dir / "Test.zip"
    zip_path.write_bytes(zip_buffer.getvalue())
    return zip_path


@pytest.fixture
def mock_dataset_paths(tmp_path):
    tmp_path = Path(tmp_path)

    user_config = tmp_path / "user_datasets.json"
    folder_config = tmp_path / "folder_datasets.json"

    with patch.object(ds, "_dir", str(tmp_path / "datasets" / "Test")), patch(
        "sed.core.config.USER_CONFIG_PATH",
        tmp_path,
    ), patch.object(
        dm,
        "json_path",
        {"user": str(user_config), "module": json_path, "folder": str(folder_config)},
    ):
        yield {"user": user_config, "folder": folder_config, "tmp_path": tmp_path}


@pytest.fixture
def sample_dataset_config():
    return {
        "Test": {
            "url": "http://test.com/files/file.zip",
            "subdirs": ["subdir"],
            "data_path": [],
            "files": [],
        },
        "TestSimple": {"url": "http://test.com/simple.zip", "data_path": [], "files": []},
    }


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


def test_set_root_dir(mock_dataset_paths, sample_dataset_config):
    """Test _set_data_dir with proper path mocking."""
    user_config = mock_dataset_paths["user"]
    tmp_path = mock_dataset_paths["tmp_path"]

    # Write sample config to temporary file
    user_config.write_text(json.dumps(sample_dataset_config))

    with patch.object(ds, "_datasets", sample_dataset_config):
        # test with existing data path
        ds.data_name = "Test"
        test_data_path = str(tmp_path / "test" / "data")
        ds._state["data_path"] = [test_data_path]
        ds._set_data_dir(root_dir=test_data_path, use_existing=True)
        assert os.path.abspath(test_data_path + "/") == ds._dir

        # test without existing data path
        ds._state["data_path"] = []
        ds._set_data_dir(root_dir=str(tmp_path / "test" / "data"), use_existing=True)
        expected_dir = str(tmp_path / "test" / "data" / "datasets" / "Test")
        assert os.path.abspath(expected_dir) == ds._dir

        # Additional tests using temporary paths
        with patch("os.getcwd", return_value=str(tmp_path)):
            ds._set_data_dir(root_dir=None, use_existing=True)
            expected_mock_dir = str(tmp_path / "datasets" / "Test")
            assert ds._dir == expected_mock_dir

        # Test with different provided path vs existing path
        test_data_path1 = str(tmp_path / "test" / "data1")
        test_data_path2 = str(tmp_path / "test" / "data2")
        ds._state["data_path"] = [test_data_path1]
        ds._set_data_dir(root_dir=test_data_path2, use_existing=True)
        assert os.path.abspath(test_data_path1 + "/") == ds._dir
        ds._set_data_dir(root_dir=test_data_path2, use_existing=False)
        expected_dir = str(tmp_path / "test" / "data2" / "datasets" / "Test")
        assert os.path.abspath(expected_dir) == ds._dir


def test_get_file_list(tmp_path):
    tmp_path = Path(tmp_path)

    test_dir = tmp_path / "test" / "data"
    test_dir.mkdir(parents=True, exist_ok=True)

    (test_dir / "file.txt").write_text("content")
    (test_dir / "file.zip").write_text("zip content")

    subdir = test_dir / "subdir"
    subdir.mkdir(exist_ok=True)
    (subdir / "file.txt").write_text("content")
    (subdir / "file.zip").write_text("zip content")

    ds._dir = str(test_dir)
    assert sorted(ds._get_file_list()) == ["file.txt", "subdir/file.txt"]
    assert sorted(ds._get_file_list(ignore_zip=False)) == [
        "file.txt",
        "file.zip",
        "subdir/file.txt",
        "subdir/file.zip",
    ]


def test_download_data(
    tmp_path,
    requests_mock,
    zip_buffer,
    mock_dataset_paths,
    sample_dataset_config,
):
    tmp_path = Path(tmp_path)

    user_config = mock_dataset_paths["user"]
    user_config.write_text(json.dumps(sample_dataset_config))

    test_dir = tmp_path / "test"
    test_dir.mkdir(exist_ok=True)

    data_url = "http://test.com/files/file.zip"
    requests_mock.get(data_url, content=zip_buffer.getvalue())

    with patch.object(ds, "_datasets", sample_dataset_config):
        ds._data_name = "Test"
        ds._state = {"data_path": []}
        ds._set_data_dir(root_dir=str(test_dir), use_existing=True)
        ds._download_data(data_url)

        expected_path = test_dir / "datasets" / "Test" / "Test.zip"
        assert expected_path.exists()

    # assert not ds._download_data("data", "test/data/", data_url)  # already exists


def test_extract_data(tmp_path, zip_buffer, mock_dataset_paths, sample_dataset_config):
    """Test extraction with proper isolation."""
    tmp_path = Path(tmp_path)

    user_config = mock_dataset_paths["user"]
    user_config.write_text(json.dumps(sample_dataset_config))

    test_dir = tmp_path / "test" / "datasets" / "Test"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create zip file in test directory
    zip_path = test_dir / "Test.zip"
    zip_path.write_bytes(zip_buffer.getvalue())

    with patch.object(ds, "_datasets", sample_dataset_config):
        ds._data_name = "Test"
        ds._dir = str(test_dir)
        ds._extract_data()

        assert (test_dir / "test_file.txt").exists()
        assert (test_dir / "subdir" / "test_subdir.txt").exists()


def test_rearrange_data(tmp_path, zip_buffer, mock_dataset_paths, sample_dataset_config):
    """Test rearrangement with proper isolation."""
    tmp_path = Path(tmp_path)

    user_config = mock_dataset_paths["user"]
    user_config.write_text(json.dumps(sample_dataset_config))

    test_dir = tmp_path / "test" / "datasets" / "Test"
    test_dir.mkdir(parents=True, exist_ok=True)

    zip_path = test_dir / "Test.zip"
    zip_path.write_bytes(zip_buffer.getvalue())

    with patch.object(ds, "_datasets", sample_dataset_config):
        ds._data_name = "Test"
        ds._dir = str(test_dir)
        ds._subdirs = ["subdir"]
        ds._extract_data()
        ds._rearrange_data()

        assert (test_dir / "test_file.txt").exists()
        assert (test_dir / "test_subdir.txt").exists()
        assert not (test_dir / "subdir").exists()

        with pytest.raises(FileNotFoundError):
            ds._subdirs = ["non_existing_subdir"]
            ds._rearrange_data()


def test_get_remove_dataset(
    tmp_path,
    requests_mock,
    zip_buffer,
    mock_dataset_paths,
    sample_dataset_config,
):
    tmp_path = Path(tmp_path)

    user_config = mock_dataset_paths["user"]
    user_config.write_text(json.dumps(sample_dataset_config))

    data_url = "http://test.com/files/file.zip"
    requests_mock.get(data_url, content=zip_buffer.getvalue())

    with patch.object(ds, "_datasets", sample_dataset_config):
        data_name = "Test"

        ds.remove(data_name)

        ds.get(data_name, root_dir=str(tmp_path), use_existing=False)

        expected_dir = tmp_path / "datasets" / "Test"
        assert ds.dir == str(expected_dir)

        # Check if subdir is removed after rearranging (if subdirs are configured)
        assert not (expected_dir / "subdir").exists()

        # Check datasets file to now have data_path listed
        datasets_json = json.loads(user_config.read_text())
        assert datasets_json[data_name]["data_path"]
        assert datasets_json[data_name]["files"]
        ds.remove(data_name)
        assert not expected_dir.exists()

        ds.get(data_name, root_dir=str(tmp_path), use_existing=False)
        ds.get(data_name, root_dir=str(tmp_path), use_existing=False)

        if hasattr(ds, "existing_data_paths") and ds.existing_data_paths:
            ds.remove(data_name, ds.existing_data_paths[0])
        else:
            ds.remove(data_name)


def test_datasets_manager(mock_dataset_paths):
    """Test dataset manager with proper isolation."""
    user_config = mock_dataset_paths["user"]
    folder_config = mock_dataset_paths["folder"]

    dm.add(
        "Test_DM",
        {"url": "http://test.com/files/file.zip", "subdirs": ["subdir"]},
        levels=["folder", "user"],
    )

    # Check configurations were written to temporary files
    folder_data = json.loads(folder_config.read_text())
    assert "Test_DM" in folder_data

    user_data = json.loads(user_config.read_text())
    assert "Test_DM" in user_data

    # Test removal
    dm.remove("Test_DM", levels=["folder"])
    folder_data = json.loads(folder_config.read_text())
    assert "Test_DM" not in folder_data

    datasets_json = json.load(open(dm.json_path["user"]))
    assert datasets_json["Test_DM"]

    dm.remove("Test_DM", levels=["user"])
    datasets_json = json.load(open(dm.json_path["user"]))
    with pytest.raises(KeyError):
        datasets_json["Test_DM"]

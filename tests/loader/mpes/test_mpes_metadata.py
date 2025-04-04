"""Tests specific for Mpes loader metadata retrieval"""
from __future__ import annotations

import datetime
import json
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

from sed.loader.mpes.metadata import get_archiver_data
from sed.loader.mpes.metadata import MetadataRetriever
from tests.test_config import mock_env_file  # noqa: F401


@pytest.fixture
def metadata_config():
    return {
        "elab_url": "http://example.com",
        "epics_pvs": ["channel1"],
        "archiver_url": "http://archiver.example.com",
        "aperture_config": {
            datetime.datetime.fromisoformat("2023-01-01T00:00:00"): {
                "fa_size": {"1.0": [(0, 1), (0, 1)]},
                "ca_size": {"1.0": (0, 1)},
            },
        },
        "lens_mode_config": {"mode1": {"lens1": 1.0, "lens2": 2.0}},
        "fa_in_channel": "fa_in",
        "fa_hor_channel": "fa_hor",
        "ca_in_channel": "ca_in",
    }


@pytest.fixture
def metadata_retriever(metadata_config, mock_env_file):  # noqa: ARG001
    return MetadataRetriever(metadata_config, "dummy_token")


def test_metadata_retriever_init(metadata_retriever):
    assert metadata_retriever.token == "dummy_token"
    assert metadata_retriever.url == "http://example.com"


def test_metadata_retriever_no_token(metadata_config, tmp_path, monkeypatch):
    monkeypatch.setattr("sed.core.config.ENV_DIR", tmp_path / ".dummy_env")
    monkeypatch.setattr("sed.core.config.SYSTEM_CONFIG_PATH", tmp_path / "system")
    monkeypatch.setattr("sed.core.config.USER_CONFIG_PATH", tmp_path / "user")
    retriever = MetadataRetriever(metadata_config)
    assert retriever.token is None

    metadata = {}
    runs = ["run1"]
    updated_metadata = retriever.fetch_elab_metadata(runs, metadata)
    assert updated_metadata == metadata


def test_metadata_retriever_no_url(metadata_config, mock_env_file):  # noqa: ARG001
    metadata_config.pop("elab_url")
    retriever = MetadataRetriever(metadata_config, "dummy_token")
    assert retriever.url is None

    metadata = {}
    runs = ["run1"]
    updated_metadata = retriever.fetch_elab_metadata(runs, metadata)
    assert updated_metadata == metadata


@patch("sed.loader.mpes.metadata.urlopen")
def test_get_archiver_data(mock_urlopen):
    """Test get_archiver_data using a mock of urlopen."""
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(
        [{"data": [{"secs": 1, "nanos": 500000000, "val": 10}]}],
    )
    mock_urlopen.return_value.__enter__.return_value = mock_response

    ts_from = datetime.datetime(2023, 1, 1).timestamp()
    ts_to = datetime.datetime(2023, 1, 2).timestamp()
    archiver_url = "http://archiver.example.com"
    archiver_channel = "channel1"

    secs, vals = get_archiver_data(archiver_url, archiver_channel, ts_from, ts_to)

    assert np.array_equal(secs, np.array([1.5]))
    assert np.array_equal(vals, np.array([10]))


@patch("sed.loader.mpes.metadata.get_archiver_data")
def test_fetch_epics_metadata(mock_get_archiver_data, metadata_retriever):
    """Test fetch_epics_metadata using a mock of get_archiver_data."""
    mock_get_archiver_data.return_value = (np.array([1.5]), np.array([10]))
    metadata = {"file": {}}
    ts_from = datetime.datetime(2023, 1, 1).timestamp()
    ts_to = datetime.datetime(2023, 1, 2).timestamp()

    updated_metadata = metadata_retriever.fetch_epics_metadata(ts_from, ts_to, metadata)

    assert updated_metadata["file"]["channel1"] == 10


@patch("sed.loader.mpes.metadata.get_archiver_data")
def test_fetch_epics_metadata_missing_channels(mock_get_archiver_data, metadata_retriever):
    """Test fetch_epics_metadata with missing EPICS channels."""
    mock_get_archiver_data.return_value = (np.array([1.5]), np.array([10]))
    metadata = {"file": {"channel1": 10}}
    ts_from = datetime.datetime(2023, 1, 1).timestamp()
    ts_to = datetime.datetime(2023, 1, 2).timestamp()

    updated_metadata = metadata_retriever.fetch_epics_metadata(ts_from, ts_to, metadata)

    assert "channel1" in updated_metadata["file"]


@patch("sed.loader.mpes.metadata.get_archiver_data")
def test_fetch_epics_metadata_missing_aperture_config(mock_get_archiver_data, metadata_retriever):
    """Test fetch_epics_metadata with missing aperture configuration."""
    mock_get_archiver_data.return_value = (np.array([1.5]), np.array([10]))
    metadata = {"file": {}}
    ts_from = datetime.datetime(2023, 1, 1).timestamp()
    ts_to = datetime.datetime(2023, 1, 2).timestamp()
    metadata_retriever._config["aperture_config"] = {}

    updated_metadata = metadata_retriever.fetch_epics_metadata(ts_from, ts_to, metadata)

    assert "instrument" in updated_metadata


@patch("sed.loader.mpes.metadata.get_archiver_data")
def test_fetch_epics_metadata_missing_field_aperture(mock_get_archiver_data, metadata_retriever):
    """Test fetch_epics_metadata with missing field aperture shape and size."""
    mock_get_archiver_data.return_value = (np.array([1.5]), np.array([10]))
    metadata = {"file": {}}
    ts_from = datetime.datetime(2023, 1, 1).timestamp()
    ts_to = datetime.datetime(2023, 1, 2).timestamp()

    updated_metadata = metadata_retriever.fetch_epics_metadata(ts_from, ts_to, metadata)

    assert updated_metadata["instrument"]["analyzer"]["fa_shape"] == "circle"
    assert updated_metadata["instrument"]["analyzer"]["ca_shape"] == "circle"
    assert np.isnan(updated_metadata["instrument"]["analyzer"]["fa_size"])
    assert np.isnan(updated_metadata["instrument"]["analyzer"]["ca_size"])


@patch("sed.loader.mpes.metadata.elabapi_python")
def test_fetch_elab_metadata(mock_elabapi_python, metadata_config, mock_env_file):  # noqa: ARG001
    """Test fetch_elab_metadata using a mock of elabapi_python."""
    mock_experiment = MagicMock()
    mock_experiment.id = 1
    mock_experiment.userid = 1
    mock_experiment.title = "Test Experiment"
    mock_experiment.body = "Test Body"
    mock_experiment.metadata = json.dumps({"extra_fields": {"key": {"value": "value"}}})
    mock_elabapi_python.ExperimentsApi.return_value.read_experiments.return_value = [
        mock_experiment,
    ]
    mock_user = MagicMock()
    mock_user.fullname = "Test User"
    mock_user.email = "test@example.com"
    mock_user.userid = 1
    mock_user.orcid = "0000-0000-0000-0000"
    mock_elabapi_python.UsersApi.return_value.read_user.return_value = mock_user
    mock_elabapi_python.LinksToItemsApi.return_value.read_entity_items_links.return_value = []

    metadata_retriever = MetadataRetriever(metadata_config, "dummy_token")

    metadata = {}
    runs = ["run1"]

    updated_metadata = metadata_retriever.fetch_elab_metadata(runs, metadata)

    assert updated_metadata["elabFTW"]["user"]["name"] == "Test User"
    assert updated_metadata["elabFTW"]["user"]["email"] == "test@example.com"
    assert updated_metadata["elabFTW"]["user"]["id"] == 1
    assert updated_metadata["elabFTW"]["user"]["orcid"] == "0000-0000-0000-0000"
    assert updated_metadata["elabFTW"]["scan"]["title"] == "Test Experiment"
    assert updated_metadata["elabFTW"]["scan"]["summary"] == "Test Body"
    assert updated_metadata["elabFTW"]["scan"]["id"] == 1
    assert updated_metadata["elabFTW"]["scan"]["key"] == "value"

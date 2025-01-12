"""Tests for FlashLoader metadata functionality"""
from __future__ import annotations

from pathlib import Path

import pytest

from sed.loader.flash.metadata import MetadataRetriever


@pytest.fixture
def mock_requests(requests_mock) -> None:
    # Mocking the response for the dataset URL
    dataset_url = "https://example.com/Datasets/11013410%2F43878"
    requests_mock.get(dataset_url, json={"fake": "data"}, status_code=200)


@pytest.fixture
def mock_env_token(monkeypatch, tmp_path) -> None:
    # Create a temporary .env file
    env_path = tmp_path / ".env"
    env_path.write_text("SCICAT_TOKEN=env_test_token")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)


def test_get_metadata_with_explicit_token(mock_requests: None) -> None:  # noqa: ARG001
    metadata_config = {
        "archiver_url": "https://example.com",
    }
    retriever = MetadataRetriever(metadata_config, token="explicit_test_token")
    metadata = retriever.get_metadata("11013410", ["43878"])
    assert isinstance(metadata, dict)
    assert metadata == {"fake": "data"}


def test_get_metadata_with_config_token(mock_requests: None) -> None:  # noqa: ARG001
    metadata_config = {
        "archiver_url": "https://example.com",
        "token": "config_test_token",
    }
    retriever = MetadataRetriever(metadata_config)
    metadata = retriever.get_metadata("11013410", ["43878"])
    assert isinstance(metadata, dict)
    assert metadata == {"fake": "data"}


def test_get_metadata_with_env_token(mock_requests: None, mock_env_token: None) -> None:  # noqa: ARG001
    metadata_config = {
        "archiver_url": "https://example.com",
    }
    retriever = MetadataRetriever(metadata_config)
    metadata = retriever.get_metadata("11013410", ["43878"])
    assert isinstance(metadata, dict)
    assert metadata == {"fake": "data"}


def test_get_metadata_no_token() -> None:
    metadata_config = {
        "archiver_url": "https://example.com",
    }
    with pytest.raises(ValueError, match="Token is required for metadata collection"):
        MetadataRetriever(metadata_config)


def test_get_metadata_no_url() -> None:
    metadata_config: dict = {}
    with pytest.raises(ValueError, match="No URL provided for fetching metadata"):
        MetadataRetriever(metadata_config, token="test_token")


def test_get_metadata_with_existing_metadata(mock_requests: None) -> None:  # noqa: ARG001
    metadata_config = {
        "archiver_url": "https://example.com",
        "token": "test_token",
    }
    retriever = MetadataRetriever(metadata_config)
    existing_metadata = {"existing": "metadata"}
    metadata = retriever.get_metadata("11013410", ["43878"], existing_metadata)
    assert isinstance(metadata, dict)
    assert metadata == {"existing": "metadata", "fake": "data"}


def test_get_metadata_per_run(mock_requests: None) -> None:  # noqa: ARG001
    metadata_config = {
        "archiver_url": "https://example.com",
        "token": "test_token",
    }
    retriever = MetadataRetriever(metadata_config)
    metadata = retriever._get_metadata_per_run("11013410/43878")
    assert isinstance(metadata, dict)
    assert metadata == {"fake": "data"}


def test_create_dataset_url_by_PID() -> None:
    metadata_config = {
        "archiver_url": "https://example.com",
        "token": "test_token",
    }
    retriever = MetadataRetriever(metadata_config)
    pid = "11013410/43878"
    url = retriever._create_new_dataset_url(pid)
    expected_url = "https://example.com/Datasets/11013410%2F43878"
    assert isinstance(url, str)
    assert url == expected_url

import pytest

from sed.loader.flash.metadata import MetadataRetriever


@pytest.fixture
def mock_requests(requests_mock):
    # Mocking the response for the dataset URL
    dataset_url = "https://example.com/Datasets/11013410%2F43878"
    requests_mock.get(dataset_url, json={"fake": "data"}, status_code=200)


# Test cases for MetadataRetriever
def test_get_metadata(mock_requests):  # noqa: ARG001
    metadata_config = {
        "scicat_url": "https://example.com",
        "scicat_token": "fake_token",
    }
    retriever = MetadataRetriever(metadata_config)
    metadata = retriever.get_metadata("11013410", ["43878"])
    assert isinstance(metadata, dict)
    assert metadata == {"fake": "data"}


def test_get_metadata_with_existing_metadata(mock_requests):  # noqa: ARG001
    metadata_config = {
        "scicat_url": "https://example.com",
        "scicat_token": "fake_token",
    }
    retriever = MetadataRetriever(metadata_config)
    existing_metadata = {"existing": "metadata"}
    metadata = retriever.get_metadata("11013410", ["43878"], existing_metadata)
    assert isinstance(metadata, dict)
    assert metadata == {"existing": "metadata", "fake": "data"}


def test_get_metadata_per_run(mock_requests):  # noqa: ARG001
    metadata_config = {
        "scicat_url": "https://example.com",
        "scicat_token": "fake_token",
    }
    retriever = MetadataRetriever(metadata_config)
    metadata = retriever._get_metadata_per_run("11013410/43878")
    assert isinstance(metadata, dict)
    assert metadata == {"fake": "data"}


def test_create_dataset_url_by_PID():
    metadata_config = {
        "scicat_url": "https://example.com",
        "scicat_token": "fake_token",
    }
    retriever = MetadataRetriever(metadata_config)
    # Assuming the dataset follows the new format
    pid = "11013410/43878"
    url = retriever._create_new_dataset_url(pid)
    expected_url = "https://example.com/Datasets/11013410%2F43878"
    assert isinstance(url, str)
    assert url == expected_url

from unittest.mock import patch

import pytest

from sed.loader.cfel.loader import CFELLoader

# Dummy config
config = {
    "core": {
        "instrument": "hextof",
        "beamtime_id": "12345",
        "year": "2024",
        "beamline": "pg2",
        "beamtime_dir": {"pg2": "/tmp/beamtime"},
        "paths": {"raw": "/tmp/raw"},
    },
    "dataframe": {"daq": "fadc"},
    "metadata": {"scicat_url": "http://fake.url"},
}


@pytest.fixture
def loader():
    return CFELLoader(config=config)


def test_parse_scicat_metadata(loader):
    with patch("sed.loader.cfel.loader.MetadataRetriever") as MockRetriever:
        instance = MockRetriever.return_value
        instance.get_metadata.return_value = {"scientificMetadata": {"key": "value"}}

        loader.runs = ["1"]
        meta = loader.parse_scicat_metadata(token="fake_token")

        assert meta == {"scientificMetadata": {"key": "value"}}
        instance.get_metadata.assert_called_once_with(
            beamtime_id="12345",
            runs=["1"],
            metadata={},
        )


def test_parse_local_metadata(loader):
    with patch("sed.loader.cfel.loader.MetadataRetriever") as MockRetriever:
        instance = MockRetriever.return_value
        instance.get_local_metadata.return_value = {"local": "meta"}

        loader.runs = ["1"]
        # Mock paths since _initialize_dirs might not be called or fail
        loader.beamtime_dir = "/tmp/bt"
        loader.meta_dir = "/tmp/meta"

        meta = loader.parse_local_metadata()

        assert meta == {"local": "meta"}
        instance.get_local_metadata.assert_called_once_with(
            beamtime_id="12345",
            beamtime_dir="/tmp/bt",
            meta_dir="/tmp/meta",
            runs=["1"],
            metadata={},
        )

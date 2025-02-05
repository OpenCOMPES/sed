"""This code  performs several tests for the metadata handler module.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from sed.core.metadata import DuplicateEntryError
from sed.core.metadata import MetaHandler

metadata: dict[Any, Any] = {}
metadata["entry_title"] = "Title"
# sample
metadata["sample"] = {}
metadata["sample"]["size"] = np.array([1, 2, 3])
metadata["sample"]["name"] = "Sample Name"


@pytest.fixture
def meta_handler():
    # Create a MetaHandler instance
    return MetaHandler(meta=metadata)


def test_add_entry_overwrite(meta_handler):
    # Add a new entry to metadata with 'overwrite' policy
    new_entry = {"sample": "Sample Name"}
    meta_handler.add(new_entry, "sample", duplicate_policy="overwrite")
    assert "sample" in meta_handler.metadata
    assert meta_handler.metadata["sample"] == new_entry


def test_add_entry_raise(meta_handler):
    # Attempt to add a duplicate entry with 'raise' policy
    with pytest.raises(DuplicateEntryError):
        meta_handler.add({}, "entry_title", duplicate_policy="raise")


def test_add_entry_append(meta_handler):
    # Add a new entry to metadata with 'append' policy
    new_entry = {"sample": "Sample Name"}
    meta_handler.add(new_entry, "sample", duplicate_policy="append")
    assert "sample" in meta_handler.metadata
    assert "sample_1" in meta_handler.metadata
    assert meta_handler.metadata["sample_1"] == new_entry


def test_add_entry_merge(meta_handler):
    # Add a new entry to metadata with 'merge' policy
    entry_to_merge = {"name": "Name", "type": "type"}
    meta_handler.add(entry_to_merge, "sample", duplicate_policy="merge")
    print(meta_handler.metadata)
    assert "sample" in meta_handler.metadata
    assert "name" in meta_handler.metadata["sample"]
    assert "type" in meta_handler.metadata["sample"]


def test_repr(meta_handler):
    # Test the __repr__ method
    assert repr(meta_handler) == json.dumps(metadata, default=str, indent=4)


def test_repr_html(meta_handler):
    # Test the _repr_html_ method
    html = meta_handler._format_attributes(metadata)
    assert meta_handler._repr_html_() == html

    html_test = "<div style='padding-left: 0px;'><b>Entry Title</b> [entry_title]: Title</div>"
    html_test += (
        "<div style='padding-left: 0px;'><details><summary><b>Sample</b> [sample]</summary>"
    )
    html_test += "<div style='padding-left: 20px;'><b>Size</b> [size]: (3,)</div>"
    html_test += "<div style='padding-left: 20px;'><b>Name</b> [name]: Sample Name"
    html_test += "</div></details></div>"
    assert html == html_test

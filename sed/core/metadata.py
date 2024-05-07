"""This is a metadata handler class from the sed package
"""
import json
from copy import deepcopy
from typing import Any
from typing import Dict

from sed.core.config import complete_dictionary


class MetaHandler:
    """This class provides methods to manipulate metadata dictionaries,
    and give a nice representation of them."""

    def __init__(self, meta: Dict = None) -> None:
        self._m = deepcopy(meta) if meta is not None else {}

    def __getitem__(self, val: Any) -> None:
        return self._m[val]

    def __repr__(self) -> str:
        return json.dumps(self._m, default=str, indent=4)

    def _format_attributes(self, attributes, indent=0):
        INDENT_FACTOR = 20
        html = ""
        for key, value in attributes.items():
            # Format key
            formatted_key = key.replace("_", " ").title()
            formatted_key = f"<b>{formatted_key}</b>"

            html += f"<div style='padding-left: {indent * INDENT_FACTOR}px;'>"
            if isinstance(value, dict):
                html += f"<details><summary>{formatted_key} [{key}]</summary>"
                html += self._format_attributes(value, indent + 1)
                html += "</details>"
            elif hasattr(value, "shape"):
                html += f"{formatted_key} [{key}]: {value.shape}"
            else:
                html += f"{formatted_key} [{key}]: {value}"
            html += "</div>"
        return html

    def _repr_html_(self) -> str:
        html = self._format_attributes(self._m)
        return html

    @property
    def metadata(self) -> Dict:
        """Property returning the metadata dict.
        Returns:
            dict: Dictionary of metadata.
        """
        return self._m

    def add(
        self,
        entry: Any,
        name: str,
        duplicate_policy: str = "raise",
    ) -> None:
        """Add an entry to the metadata container

        Args:
            entry: dictionary containing the metadata to add.
            name: name of the dictionary key under which to add entry.
            duplicate_policy: Control behaviour in case the 'name' key
                is already present in the metadata dictionary. Can be any of:

                    - "raise": raises a DuplicateEntryError.
                    - "overwrite": overwrites the previous data with the new one.
                    - "merge": If ``entry`` is a dictionary, recursively merges it
                      into the existing one, overwriting existing entries. Otherwise
                      the same as "overwrite".
                    - "append": adds a trailing number, keeping both entries.

        Raises:
            DuplicateEntryError: Raised if an entry already exists.
        """
        if name not in self._m.keys() or duplicate_policy == "overwrite":
            self._m[name] = deepcopy(entry)
        elif duplicate_policy == "raise":
            raise DuplicateEntryError(
                f"an entry {name} already exists in metadata",
            )
        elif duplicate_policy == "append":
            i = 0
            while True:
                i += 1
                newname = f"{name}_{i}"
                if newname not in self._m.keys():
                    break
            self._m[newname] = deepcopy(entry)

        elif duplicate_policy == "merge":
            if isinstance(self._m[name], dict):
                if not isinstance(entry, dict):
                    raise ValueError(
                        "Cannot merge dictionary with non-dictionary entry!",
                    )
                complete_dictionary(self._m[name], deepcopy(entry))
            else:
                self._m[name] = deepcopy(entry)

        else:
            raise ValueError(
                f"could not interpret duplication handling method {duplicate_policy}"
                f"Please choose between overwrite,append or raise.",
            )


class DuplicateEntryError(Exception):
    """Exception raised when attempting to add a duplicate entry to the metadata container.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message: str = "An entry already exists in metadata"):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.__class__.__name__}: {self.message}"

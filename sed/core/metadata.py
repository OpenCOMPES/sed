"""This is a metadata handler class from the sed package

"""
from copy import deepcopy
from typing import Any
from typing import Dict

from sed.core.config import complete_dictionary


class MetaHandler:
    """[summary]"""

    def __init__(self, meta: Dict = None) -> None:
        self._m = deepcopy(meta) if meta is not None else {}

    def __getitem__(self, val: Any) -> None:
        return self._m[val]

    def __repr__(self) -> str:
        # TODO: #35 add pretty print, possibly to HTML
        return str(self._m)

    @property
    def metadata(self) -> dict:
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

    def add_processing(self, method: str, **kwds: Any) -> None:
        """docstring

        Args:

        Returns:

        """
        # TODO: #36 Add processing metadata validation tests
        self._m["processing"][method] = kwds

    def from_nexus(self, val: Any) -> None:
        """docstring

        Args:

        Returns:

        """
        raise NotImplementedError()

    def to_nexus(self, val: Any) -> None:
        """docstring

        Args:

        Returns:

        """
        raise NotImplementedError()

    def from_json(self, val: Any) -> None:
        """docstring

        Args:

        Returns:

        """
        raise NotImplementedError()

    def to_json(self, val: Any) -> None:
        """docstring

        Args:

        Returns:

        """
        raise NotImplementedError()

    def from_dict(self, val: Any) -> None:
        """docstring

        Args:

        Returns:

        """
        raise NotImplementedError()

    def to_dict(self, val: Any) -> None:
        """docstring

        Args:

        Returns:

        """
        raise NotImplementedError()


class DuplicateEntryError(Exception):
    """[summary]"""


if __name__ == "__main__":
    m = MetaHandler()
    m.add({"start": 0, "stop": 1}, name="test")
    print(m)

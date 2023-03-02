"""This is a metadata handler class from the sed package

"""
from typing import Any
from typing import Dict


class MetaHandler:
    """[summary]"""

    def __init__(self, meta: Dict = None) -> None:
        self._m = meta if meta is not None else {}

    def __getitem__(self, val: Any) -> None:
        return self._m[val]

    def __repr__(self) -> str:
        # TODO: #35 add pretty print, possibly to HTML
        return str(self._m)

    def add(
        self,
        entry: Dict,
        category=None,
        duplicate_policy: str = "raise",
    ) -> None:
        """Add an entry to the metadata container

        Args:
            v: dictionary containing the metadata to add.
                Must contain a 'name' key.
            category: sub-dictionary container where to store the entry. e.g. "workflow"
                for workflow steps
            duplicate_policy: Control behaviour in case the 'name' key
                is already present in the metadata dictionary. If raise, raises
                a DuplicateEntryError.
                If 'overwrite' it overwrites the previous data with the new
                one.
                If 'append' it adds a trailing number, keeping both entries.

        Raises:
            DuplicateEntryError: [description]
        """
        meta = self._m
        if category is not None:
            if category not in self._m.keys():
                self._m[category] = {}
            meta = meta[category]

        if entry["name"] not in meta.keys() or duplicate_policy == "overwrite":
            meta[entry["name"]] = entry
        elif duplicate_policy == "raise":
            raise DuplicateEntryError(
                f"an entry {entry['name']} already exists in metadata",
            )
        elif duplicate_policy == "append":
            i = 0
            while True:
                i += 1
                newname = f"{entry['name']}_{i}"
                if newname not in meta.keys():
                    break
            meta[newname] = entry
        else:
            raise ValueError(
                f"could not interpret duplication handling method {duplicate_policy}"
                f"Please choose between overwrite,append or raise.",
            )

    def has_processing(self, method) -> bool:
        if method in self._m["workflow"]:
            return True
        return False

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
    m.add({"name": "test", "start": 0, "stop": 1})
    print(m)

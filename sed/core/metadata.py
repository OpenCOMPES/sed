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

    def add(self, v: Dict, category=None, duplicate: str = "raise") -> None:
        """Add an entry to the metadata container

        Args:
            v: dictionary containing the metadata to add.
                Must contain a 'name' key.
            category: sub-dictionary container where to store the entry. e.g. "workflow"
                for workflow steps
            overwrite: Control behaviour in case the 'name' key
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

        if v["name"] not in meta.keys() or duplicate == "overwrite":
            meta[v["name"]] = v
        elif duplicate == "raise":
            raise DuplicateEntryError(
                f"an entry {v['name']} already exists in metadata",
            )
        elif duplicate == "append":
            i = 0
            while True:
                i += 1
                newname = f"{v['name']}_{i}"
                if newname not in meta.keys():
                    break
            meta[newname] = v

        else:
            raise ValueError(
                f"could not interpret duplication handling method {duplicate}"
                f"Please choose between overwrite,append or raise.",
            )

    # def add_processing(self, method: str, **kwds: Any) -> None:
    #     """docstring

    #     Args:

    #     Returns:

    #     """
    #     # TODO: #36 Add processing metadata validation tests
    #     self._m["processing"][method] = kwds

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

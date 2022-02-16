from typing import Any
from typing import Dict


class MetaHandler:
    def __init__(self, meta: Dict = None) -> None:
        self._m = meta if meta is not None else {}

    def __getitem__(self, val: Any) -> None:
        return self._m[val]

    def __repr__(self) -> str:
        # TODO: #35 add pretty print, possibly to HTML
        return str(self._m)

    def add(self, v: Dict, duplicate: bool = "raise") -> None:
        """Add an entry to the metadata container

        Args:
            v (Dict): dictionary containing the metadata to add.
                Must contain a 'name' key.
            overwrite (str, optional): Control behaviour in case the 'name' key
                is already present in the metadata dictionary. If raise, raises
                a DuplicateEntryError.
                If 'overwrite' it overwrites the previous data with the new
                one.
                If 'append' it adds a trailing number, keeping both entries.
                Defaults to 'raise'.

        Raises:
            DuplicateEntryError: [description]
        """
        if v["name"] not in self._m.keys() or duplicate == "overwrite":
            self._m[v["name"]] = v
        elif duplicate == "raise":
            raise DuplicateEntryError(
                f"an entry {v['name']} already exists in metadata",
            )
        elif duplicate == "append":
            n = 0
            while True:
                n += 1
                newname = f"name_{n}"
                if newname not in self._m.keys():
                    break
            self._m[newname] = v

        else:
            raise ValueError(
                f"could not interpret duplication handling method {duplicate}"
                f"Please choose between overwrite,append or rise.",
            )

    def addProcessing(self, method: str, **kwds: Any) -> None:
        # TODO: #36 Add processing metadata validation tests
        self._m["processing"][method] = kwds

    def from_nexus(self, val: Any) -> None:
        raise NotImplementedError()

    def to_nexus(self, val: Any) -> None:
        raise NotImplementedError()

    def from_json(self, val: Any) -> None:
        raise NotImplementedError()

    def to_json(self, val: Any) -> None:
        raise NotImplementedError()

    def from_dict(self, val: Any) -> None:
        raise NotImplementedError()

    def to_dict(self, val: Any) -> None:
        raise NotImplementedError()


class DuplicateEntryError(Exception):
    pass


if __name__ == "__main__":
    m = MetaHandler()
    m.add({"name": "test", "start": 0, "stop": 1})
    print(m)

from enum import Enum
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

from pydantic import BaseModel
from pydantic import DirectoryPath
from pydantic import field_validator
from pydantic import model_validator


class DataFormat(str, Enum):
    PER_ELECTRON = "per_electron"
    PER_PULSE = "per_pulse"
    PER_TRAIN = "per_train"


class DataPaths(BaseModel):
    """
    Represents paths for raw and parquet data in a beamtime directory.
    """

    data_raw_dir: DirectoryPath
    data_parquet_dir: DirectoryPath

    @field_validator("data_parquet_dir", mode="before")
    @classmethod
    def check_and_create_parquet_dir(cls, v):
        v = Path(v)
        if not v.is_dir():
            v.mkdir(parents=True, exist_ok=True)
        return v

    @classmethod
    def from_beamtime_dir(
        cls,
        loader: str,
        beamtime_dir: Path,
        beamtime_id: int,
        year: int,
        daq: str,
    ) -> "DataPaths":
        """
        Create DataPaths instance from a beamtime directory and DAQ type.

        Parameters:
        - beamtime_dir (Path): Path to the beamtime directory.
        - daq (str): Type of DAQ.

        Returns:
        - DataPaths: Instance of DataPaths.
        """
        data_raw_dir_list = []
        if loader == "flash":
            beamtime_dir = beamtime_dir.joinpath(f"{year}/data/{beamtime_id}/")
            raw_path = beamtime_dir.joinpath("raw")

            for path in raw_path.glob("**/*"):
                if path.is_dir():
                    dir_name = path.name
                    if dir_name.startswith("express-") or dir_name.startswith(
                        "online-",
                    ):
                        data_raw_dir_list.append(path.joinpath(daq))
                    elif dir_name == daq.upper():
                        data_raw_dir_list.append(path)
            data_raw_dir = data_raw_dir_list[0]

        if loader == "sxp":
            beamtime_dir = beamtime_dir.joinpath(f"{year}/{beamtime_id}/")
            data_raw_dir = beamtime_dir.joinpath("raw")

        if not data_raw_dir.is_dir():
            raise FileNotFoundError("Raw data directories not found.")

        parquet_path = "processed/parquet"
        data_parquet_dir = beamtime_dir.joinpath(parquet_path)
        data_parquet_dir.mkdir(parents=True, exist_ok=True)

        return cls(data_raw_dir=data_raw_dir, data_parquet_dir=data_parquet_dir)


class AuxiliaryChannel(BaseModel):
    """
    Represents auxiliary channels in DLD.
    """

    name: str
    slice: int
    dtype: Optional[str] = None


class Channel(BaseModel):
    """
    Represents a data channel.
    """

    name: str
    format: DataFormat
    group_name: Optional[str] = None
    index_key: Optional[str] = None
    dataset_key: Optional[str] = None
    slice: Optional[int] = None
    dtype: Optional[str] = None
    dldAuxChannels: Optional[dict] = None
    max_hits: Optional[int] = None
    scale: Optional[float] = None

    @model_validator(mode="after")
    def set_index_dataset_key(self):
        if self.index_key and self.dataset_key:
            return self
        if self.group_name:
            self.index_key = self.group_name + "index"
            if self.name == "timeStamp":
                self.dataset_key = self.group_name + "time"
            else:
                self.dataset_key = self.group_name + "value"
        else:
            raise ValueError(
                "Channel:",
                self.name,
                "Either 'group_name' or 'index_key' AND 'dataset_key' must be provided.",
            )
        return self

    # if name is dldAux, check format to be per_train. If per_pulse, correct to per_train
    @model_validator(mode="after")
    def dldAux_format(self):
        if self.name == "dldAux":
            if self.format != DataFormat.PER_TRAIN:
                print("The correct format for dldAux is per_train, not per_pulse. Correcting.")
                self.format = DataFormat.PER_TRAIN
        return self

    # validate dldAuxChannels
    @model_validator(mode="after")
    def check_dldAuxChannels(self):
        if self.name == "dldAux":
            if self.dldAuxChannels is None:
                raise ValueError(f"Channel 'dldAux' requires 'dldAuxChannels' to be defined.")
            for name, data in self.dldAuxChannels.items():
                # if data is int, convert to dict
                if isinstance(data, int):
                    self.dldAuxChannels[name] = AuxiliaryChannel(name=name, slice=data)
                elif isinstance(data, dict):
                    self.dldAuxChannels[name] = AuxiliaryChannel(name=name, **data)

        return self


class DataFrameConfig(BaseModel):
    """
    Represents configuration for DataFrame.
    """

    daq: str
    ubid_offset: int
    forward_fill_iterations: int = 2
    split_sector_id_from_dld_time: bool = False
    sector_id_reserved_bits: Optional[int] = None
    channels: Dict[str, Any]
    units: Dict[str, str] = None
    stream_name_prefixes: Dict[str, str] = None
    stream_name_prefix: str = ""
    stream_name_postfixes: Dict[str, str] = None
    stream_name_postfix: str = ""
    beamtime_dir: Dict[str, str]
    sector_id_column: Optional[str] = None
    tof_column: Optional[str] = "dldTimeSteps"
    num_trains: Optional[int] = None

    @field_validator("channels", mode="before")
    @classmethod
    def populate_channels(cls, v):
        return {name: Channel(name=name, **data) for name, data in v.items()}

    # validate that pulseId
    @field_validator("channels", mode="after")
    @classmethod
    def check_channels(cls, v):
        if "pulseId" not in v:
            raise ValueError("Channel: pulseId must be provided.")
        return v

    # valide split_sector_id_from_dld_time and sector_id_reserved_bits
    @model_validator(mode="after")
    def check_sector_id_reserved_bits(self):
        if self.split_sector_id_from_dld_time:
            if self.sector_id_reserved_bits is None:
                raise ValueError(
                    "'split_sector_id_from_dld_time' is True",
                    "Please provide 'sector_id_reserved_bits'.",
                )
            if self.sector_id_column is None:
                print("No sector_id_column provided. Defaulting to dldSectorID.")
                self.sector_id_column = "dldSectorID"
        return self

    # compute stream_name_prefix and stream_name_postfix
    @model_validator(mode="after")
    def set_stream_name_prefix_and_postfix(self):
        if self.stream_name_prefixes is not None:
            # check if daq is in stream_name_prefixes
            if self.daq not in self.stream_name_prefixes:
                raise ValueError(
                    f"DAQ type '{self.daq}' not found in stream_name_prefixes.",
                )
            self.stream_name_prefix = self.stream_name_prefixes[self.daq]

        if self.stream_name_postfixes is not None:
            # check if daq is in stream_name_postfixes
            if self.daq not in self.stream_name_postfixes:
                raise ValueError(
                    f"DAQ type '{self.daq}' not found in stream_name_postfixes.",
                )
            self.stream_name_postfix = self.stream_name_postfixes[self.daq]

        return self


class CoreConfig(BaseModel):
    """
    Represents core configuration for Flash.
    """

    loader: str = None
    beamline: str = None
    paths: Optional[DataPaths] = None
    beamtime_id: int = None
    year: int = None
    base_folder: Optional[str] = None


class LoaderConfig(BaseModel):
    """
    Configuration for the flash loader.
    """

    core: CoreConfig
    dataframe: DataFrameConfig
    metadata: Optional[Dict] = None
    nexus: Optional[Dict] = None

    @model_validator(mode="after")
    def check_paths(self):
        if self.core.paths is None:
            # check if beamtime_id and year are set
            if self.core.beamtime_id is None or self.core.year is None:
                raise ValueError("Either 'paths' or 'beamtime_id' and 'year' must be provided.")

            daq = self.dataframe.daq
            beamtime_dir_path = Path(self.dataframe.beamtime_dir[self.core.beamline])
            self.core.paths = DataPaths.from_beamtime_dir(
                self.core.loader,
                beamtime_dir_path,
                self.core.beamtime_id,
                self.core.year,
                daq,
            )

        return self

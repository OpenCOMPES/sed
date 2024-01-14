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


class DAQType(str, Enum):
    PBD = "pbd"
    PBD2 = "pbd2"
    FL1USER1 = "fl1user1"
    FL1USER2 = "fl1user2"
    FL1USER3 = "fl1user3"
    FL2USER1 = "fl2user1"
    FL2USER2 = "fl2user2"


class DataPaths(BaseModel):
    """
    Represents paths for raw and parquet data in a beamtime directory.
    """

    data_raw_dir: DirectoryPath
    data_parquet_dir: DirectoryPath

    @classmethod
    def from_beamtime_dir(
        cls,
        beamtime_dir: Path,
        beamtime_id: int,
        year: int,
        daq: DAQType,
    ) -> "DataPaths":
        """
        Create DataPaths instance from a beamtime directory and DAQ type.

        Parameters:
        - beamtime_dir (Path): Path to the beamtime directory.
        - daq (str): Type of DAQ.

        Returns:
        - DataPaths: Instance of DataPaths.
        """
        beamtime_dir = beamtime_dir.joinpath(f"{year}/data/{beamtime_id}/")
        raw_path = beamtime_dir.joinpath("raw")
        data_raw_dir = []
        for path in raw_path.glob("**/*"):
            if path.is_dir():
                dir_name = path.name
                if dir_name.startswith("express-") or dir_name.startswith(
                    "online-",
                ):
                    data_raw_dir.append(path.joinpath(daq.value))
                elif dir_name == daq.value.upper():
                    data_raw_dir.append(path)
        print(type(data_raw_dir[0]))
        if not data_raw_dir:
            raise ValueError(f"Raw data directories for DAQ type '{daq.value}' not found.")

        parquet_path = "processed/parquet"
        data_parquet_dir = beamtime_dir.joinpath(parquet_path)

        return cls(data_raw_dir=data_raw_dir[0], data_parquet_dir=data_parquet_dir)


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

    daq: DAQType
    ubid_offset: int
    forward_fill_iterations: int = 2
    split_sector_id_from_dld_time: bool = False
    sector_id_reserved_bits: Optional[int] = None
    channels: Dict[str, Any]
    units: Dict[str, str] = None
    stream_name_prefixes: Dict[str, str]
    beamtime_dir: Dict[str, str]
    sector_id_column: Optional[str] = None
    tof_column: Optional[str] = "dldTimeSteps"

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


class CoreConfig(BaseModel):
    """
    Represents core configuration for Flash.
    """

    beamline: str = None
    paths: Optional[DataPaths] = None
    beamtime_id: int = None
    year: int = None
    base_folder: Optional[str] = None


class Nexus(BaseModel):
    """
    Represents Nexus configuration.
    """

    reader: str = None
    definition: str = None
    input_files: str = None


class Metadata(BaseModel):
    """
    Represents metadata configuration.
    """

    scicat_url: str = None
    scicat_username: str = None
    scicat_password: str = None


class FlashLoaderConfig(BaseModel):
    """
    Configuration for the flash loader.
    """

    core: CoreConfig
    dataframe: DataFrameConfig
    metadata: Optional[Metadata] = None
    nexus: Optional[Nexus] = None

    @model_validator(mode="after")
    def check_paths(self):
        if self.core.paths is None:
            # check if beamtime_id and year are set
            if self.core.beamtime_id is None or self.core.year is None:
                raise ValueError("Either 'paths' or 'beamtime_id' and 'year' must be provided.")

            daq = self.dataframe.daq
            beamtime_dir_path = Path(self.dataframe.beamtime_dir[self.core.beamline])
            self.core.paths = DataPaths.from_beamtime_dir(
                beamtime_dir_path,
                self.core.beamtime_id,
                self.core.year,
                daq,
            )

        return self

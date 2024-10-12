"""Pydantic model to validate the config for SED package."""
from collections.abc import Sequence
from typing import Literal
from typing import Optional
from typing import Union

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import DirectoryPath
from pydantic import field_validator
from pydantic import FilePath
from pydantic import HttpUrl
from pydantic import NewPath
from pydantic import SecretStr

from sed.loader.loader_interface import get_names_of_all_loaders

## Best to not use futures annotations with pydantic models
## https://github.com/astral-sh/ruff/issues/5434


class Paths(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw: DirectoryPath
    processed: Union[DirectoryPath, NewPath]


class CoreModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    loader: str
    verbose: Optional[bool] = None
    paths: Optional[Paths] = None
    num_cores: Optional[int] = None
    year: Optional[int] = None
    beamtime_id: Optional[int] = None
    instrument: Optional[str] = None
    beamline: Optional[str] = None
    # TODO: move copy tool to a separate model
    use_copy_tool: Optional[bool] = None
    copy_tool_source: Optional[DirectoryPath] = None
    copy_tool_dest: Optional[DirectoryPath] = None
    copy_tool_kwds: Optional[dict] = None

    @field_validator("loader")
    @classmethod
    def validate_loader(cls, v: str) -> str:
        """Checks if the loader is one of the valid ones"""
        names = get_names_of_all_loaders()
        if v not in names:
            raise ValueError(f"Invalid loader {v}. Available loaders are: {names}")
        return v


class ColumnsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: str
    y: str
    tof: str
    tof_ns: str
    kx: str
    ky: str
    energy: str
    delay: str
    adc: str
    bias: str
    timestamp: str
    corrected_x: str
    corrected_y: str
    corrected_tof: str
    corrected_delay: Optional[str] = None
    sector_id: Optional[str] = None
    auxiliary: Optional[str] = None


class ChannelModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: Literal["per_train", "per_electron", "per_pulse", "per_file"]
    dataset_key: str
    index_key: Optional[str] = None
    slice: Optional[int] = None
    dtype: Optional[str] = None

    class subChannel(BaseModel):
        model_config = ConfigDict(extra="forbid")

        slice: int
        dtype: Optional[str] = None

    sub_channels: Optional[dict[str, subChannel]] = None


class DataframeModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    columns: ColumnsModel
    units: Optional[dict[str, str]] = None
    channels: Optional[dict[str, ChannelModel]] = None
    # other settings
    tof_binwidth: float
    tof_binning: int
    adc_binning: int
    jitter_cols: Sequence[str]
    jitter_amps: Union[float, Sequence[float]]
    timed_dataframe_unit_time: float
    # flash specific settings
    forward_fill_iterations: Optional[int] = None
    ubid_offset: Optional[int] = None
    split_sector_id_from_dld_time: Optional[bool] = None
    sector_id_reserved_bits: Optional[int] = None
    sector_delays: Optional[Sequence[float]] = None
    daq: Optional[str] = None

    # write validator for model so that x_column gets converted to columns: x


class BinningModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hist_mode: str
    mode: str
    pbar: bool
    threads_per_worker: int
    threadpool_API: str


class HistogramModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bins: Sequence[int]
    axes: Sequence[str]
    ranges: Sequence[tuple[float, float]]


class StaticModel(BaseModel):
    """Static configuration settings that shouldn't be changed by users."""

    # flash specific settings
    model_config = ConfigDict(extra="forbid")

    stream_name_prefixes: Optional[dict] = None
    stream_name_postfixes: Optional[dict] = None
    beamtime_dir: Optional[dict] = None


class EnergyModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bins: int
    ranges: Sequence[int]
    normalize: bool
    normalize_span: int
    normalize_order: int
    fastdtw_radius: int
    peak_window: int
    calibration_method: str
    energy_scale: str
    tof_fermi: int
    tof_width: Sequence[int]
    x_width: Sequence[int]
    y_width: Sequence[int]
    color_clip: int
    bias_key: Optional[str] = None

    class EnergyCalibrationModel(BaseModel):
        model_config = ConfigDict(extra="forbid")

        creation_date: Optional[float] = None
        d: Optional[float] = None
        t0: Optional[float] = None
        E0: Optional[float] = None
        energy_scale: str

    calibration: Optional[EnergyCalibrationModel] = None

    class EnergyOffsets(BaseModel):
        model_config = ConfigDict(extra="forbid")

        creation_date: Optional[float] = None
        constant: Optional[float] = None

        ## This seems rather complicated way to define offsets,
        # inconsistent in how args vs config are for add_offsets
        class OffsetColumn(BaseModel):
            weight: float
            preserve_mean: bool
            reduction: Optional[str] = None

        columns: Optional[dict[str, OffsetColumn]] = None

    offsets: Optional[EnergyOffsets] = None

    class EnergyCorrectionModel(BaseModel):
        model_config = ConfigDict(extra="forbid")

        creation_date: Optional[float] = None
        correction_type: str
        amplitude: float
        center: Sequence[float]
        gamma: Optional[float] = None
        sigma: Optional[float] = None
        diameter: Optional[float] = None
        sigma2: Optional[float] = None
        amplitude2: Optional[float] = None

    correction: Optional[EnergyCorrectionModel] = None


class MomentumModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    axes: Sequence[str]
    bins: Sequence[int]
    ranges: Sequence[Sequence[int]]
    detector_ranges: Sequence[Sequence[int]]
    center_pixel: Sequence[int]
    sigma: int
    fwhm: int
    sigma_radius: int

    class MomentumCalibrationModel(BaseModel):
        model_config = ConfigDict(extra="forbid")

        creation_date: Optional[float] = None
        kx_scale: float
        ky_scale: float
        x_center: float
        y_center: float
        rstart: float
        cstart: float
        rstep: float
        cstep: float

    calibration: Optional[MomentumCalibrationModel] = None

    class MomentumCorrectionModel(BaseModel):
        model_config = ConfigDict(extra="forbid")

        creation_date: Optional[float] = None
        feature_points: Sequence[Sequence[float]]
        rotation_symmetry: int
        include_center: bool
        use_center: bool

    correction: Optional[MomentumCorrectionModel] = None


class DelayModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adc_range: Sequence[int]
    flip_time_axis: bool
    # Group keys in the datafile
    p1_key: Optional[str] = None
    p2_key: Optional[str] = None
    p3_key: Optional[str] = None
    t0_key: Optional[str] = None

    class DelayCalibration(BaseModel):
        model_config = ConfigDict(extra="forbid")

        creation_date: Optional[float] = None
        adc_range: Sequence[int]
        delay_range: Sequence[float]
        time0: float
        delay_range_mm: Sequence[float]
        datafile: FilePath  # .h5 extension in filepath

    calibration: Optional[DelayCalibration] = None

    class DelayOffsets(BaseModel):
        model_config = ConfigDict(extra="forbid")

        creation_date: Optional[float] = None
        constant: Optional[float] = None
        flip_delay_axis: Optional[bool] = False

        ## This seems rather complicated way to define offsets,
        # inconsistent in how args vs config are for add_offsets
        class OffsetColumn(BaseModel):
            weight: float
            preserve_mean: bool
            reduction: Optional[str] = None

        columns: Optional[dict[str, OffsetColumn]] = None

    offsets: Optional[DelayOffsets] = None


class MetadataModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    archiver_url: Optional[HttpUrl] = None
    token: Optional[SecretStr] = None
    epics_pvs: Optional[Sequence[str]] = None
    fa_in_channel: Optional[str] = None
    fa_hor_channel: Optional[str] = None
    ca_in_channel: Optional[str] = None
    aperture_config: Optional[dict] = None
    lens_mode_config: Optional[dict] = None


class NexusModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reader: str  # prob good to have validation here
    # Currently only NXmpes definition is supported
    definition: Literal["NXmpes"]
    input_files: Sequence[FilePath]


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    core: CoreModel
    dataframe: DataframeModel
    energy: EnergyModel
    momentum: MomentumModel
    delay: DelayModel
    binning: BinningModel
    histogram: HistogramModel
    metadata: Optional[MetadataModel] = None
    nexus: Optional[NexusModel] = None
    static: Optional[StaticModel] = None

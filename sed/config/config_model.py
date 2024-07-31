from collections.abc import Sequence
from typing import Literal
from typing import Optional
from typing import Union

from pydantic import BaseModel
from pydantic import DirectoryPath
from pydantic import field_validator
from pydantic import FilePath
from pydantic import HttpUrl
from pydantic import NewPath
from pydantic import SecretStr
from typing_extensions import NotRequired
from typing_extensions import TypedDict

from sed.loader.loader_interface import get_names_of_all_loaders

## Best to not use futures annotations with pydantic models
## https://github.com/astral-sh/ruff/issues/5434


class Paths(BaseModel):
    raw: DirectoryPath
    processed: Union[DirectoryPath, NewPath]


class CoreModel(BaseModel):
    loader: str = "generic"
    paths: Optional[Paths] = None
    num_cores: int = 4
    year: Optional[int] = None
    beamtime_id: Optional[int] = None
    instrument: Optional[str] = None

    @field_validator("loader")
    @classmethod
    def validate_loader(cls, v: str) -> str:
        """Checks if the loader is one of the valid ones"""
        names = get_names_of_all_loaders()
        if v not in names:
            raise ValueError(f"Invalid loader {v}. Available loaders are: {names}")
        return v


class ColumnsModel(BaseModel):
    x: str = "X"
    y: str = "Y"
    tof: str = "t"
    tof_ns: str = "t_ns"
    corrected_x: str = "Xm"
    corrected_y: str = "Ym"
    corrected_tof: str = "tm"
    kx: str = "kx"
    ky: str = "ky"
    energy: str = "energy"
    delay: str = "delay"
    adc: str = "ADC"
    bias: str = "sampleBias"
    timestamp: str = "timeStamp"


class subChannelModel(TypedDict):
    slice: int
    dtype: NotRequired[str]


# Either channels is changed to not be dict or we use TypedDict
# However TypedDict can't accept default values
class ChannelModel(TypedDict):
    format: Literal["per_train", "per_electron", "per_pulse", "per_file"]
    dataset_key: str
    index_key: NotRequired[str]
    slice: NotRequired[int]
    dtype: NotRequired[str]
    sub_channels: NotRequired[dict[str, subChannelModel]]


class Dataframe(BaseModel):
    columns: ColumnsModel = ColumnsModel()
    units: Optional[dict[str, str]] = None
    # Since channels are not fixed, we use a TypedDict to represent them.
    channels: Optional[dict[str, ChannelModel]] = None

    tof_binwidth: float = 4.125e-12
    tof_binning: int = 1
    adc_binning: int = 1
    jitter_cols: Sequence[str] = ["@x", "@y", "@tof"]
    jitter_amps: Union[float, Sequence[float]] = 0.5
    timed_dataframe_unit_time: float = 0.001
    # flash specific settings
    forward_fill_iterations: Optional[int] = None
    ubid_offset: Optional[int] = None
    split_sector_id_from_dld_time: Optional[bool] = None
    sector_id_reserved_bits: Optional[int] = None
    sector_delays: Optional[Sequence[int]] = None


class BinningModel(BaseModel):
    hist_mode: str = "numba"
    mode: str = "fast"
    pbar: bool = True
    threads_per_worker: int = 4
    threadpool_API: str = "blas"


class HistogramModel(BaseModel):
    bins: Sequence[int] = [80, 80, 80]
    axes: Sequence[str] = ["@x", "@y", "@tof"]
    ranges: Sequence[Sequence[int]] = [[0, 1800], [0, 1800], [0, 150000]]


class StaticModel(BaseModel):
    """Static configuration settings that shouldn't be changed by users."""

    # flash specific settings
    stream_name_prefixes: Optional[dict] = None
    stream_name_postfixes: Optional[dict] = None
    beamtime_dir: Optional[dict] = None


class EnergyCalibrationModel(BaseModel):
    d: float
    t0: float
    E0: float
    energy_scale: str


class EnergyCorrectionModel(BaseModel):
    correction_type: str = "Lorentzian"
    amplitude: float
    center: Sequence[float]
    gamma: float
    sigma: float
    diameter: float


class EnergyModel(BaseModel):
    bins: int = 1000
    ranges: Sequence[int] = [100000, 150000]
    normalize: bool = True
    normalize_span: int = 7
    normalize_order: int = 1
    fastdtw_radius: int = 2
    peak_window: int = 7
    calibration_method: str = "lmfit"
    energy_scale: str = "kinetic"
    tof_fermi: int = 132250
    tof_width: Sequence[int] = [-600, 1000]
    x_width: Sequence[int] = [-20, 20]
    y_width: Sequence[int] = [-20, 20]
    color_clip: int = 300
    calibration: Optional[EnergyCalibrationModel] = None
    correction: Optional[EnergyCorrectionModel] = None


class MomentumCalibrationModel(BaseModel):
    kx_scale: float
    ky_scale: float
    x_center: float
    y_center: float
    rstart: float
    cstart: float
    rstep: float
    cstep: float


class MomentumCorrectionModel(BaseModel):
    feature_points: Sequence[Sequence[float]]
    rotation_symmetry: int
    include_center: bool
    use_center: bool


class MomentumModel(BaseModel):
    axes: Sequence[str] = ["@x", "@y", "@tof"]
    bins: Sequence[int] = [512, 512, 300]
    ranges: Sequence[Sequence[int]] = [[-256, 1792], [-256, 1792], [132000, 138000]]
    detector_ranges: Sequence[Sequence[int]] = [[0, 2048], [0, 2048]]
    center_pixel: Sequence[int] = [256, 256]
    sigma: int = 5
    fwhm: int = 8
    sigma_radius: int = 1
    calibration: Optional[MomentumCalibrationModel] = None
    correction: Optional[MomentumCorrectionModel] = None


class DelayModel(BaseModel):
    adc_range: Sequence[int] = [1900, 25600]
    time0: int = 0
    flip_time_axis: bool = False
    p1_key: Optional[str] = None
    p2_key: Optional[str] = None
    p3_key: Optional[str] = None


class MetadataModel(BaseModel):
    archiver_url: Optional[HttpUrl] = None
    token: Optional[SecretStr] = None
    epics_pvs: Optional[Sequence[str]] = None
    fa_in_channel: Optional[str] = None
    fa_hor_channel: Optional[str] = None
    ca_in_channel: Optional[str] = None
    aperture_config: Optional[dict] = None
    lens_mode_config: Optional[dict] = None


class NexusModel(BaseModel):
    reader: str  # prob good to have validation here
    # Currently only NXmpes definition is supported
    definition: Literal["NXmpes"]
    input_files: Sequence[FilePath]


class ConfigModel(BaseModel):
    core: CoreModel
    dataframe: Dataframe
    energy: EnergyModel
    momentum: MomentumModel
    delay: DelayModel
    binning: BinningModel
    histogram: HistogramModel
    metadata: Optional[MetadataModel] = None
    nexus: Optional[NexusModel] = None
    static: Optional[StaticModel] = None

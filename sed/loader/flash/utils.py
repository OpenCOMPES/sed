from __future__ import annotations

from pathlib import Path

# TODO: move to config
MULTI_INDEX = ["trainId", "pulseId", "electronId"]
PULSE_ALIAS = MULTI_INDEX[1]
DLD_AUX_ALIAS = "dldAux"
DLDAUX_CHANNELS = "dldAuxChannels"
FORMATS = ["per_electron", "per_pulse", "per_train"]


def get_channels(
    channel_dict: dict = None,
    formats: str | list[str] = None,
    index: bool = False,
    extend_aux: bool = False,
) -> list[str]:
    """
    Returns a list of channels associated with the specified format(s).

    Args:
        formats (str | list[str]): The desired format(s)
        ('per_pulse', 'per_electron', 'per_train', 'all').
        index (bool): If True, includes channels from the multiindex.
        extend_aux (bool): If True, includes channels from the 'dldAuxChannels' dictionary,
                else includes 'dldAux'.

    Returns:
        List[str]: A list of channels with the specified format(s).
    """
    # If 'formats' is a single string, convert it to a list for uniform processing.
    if isinstance(formats, str):
        formats = [formats]

    # If 'formats' is a string "all", gather all possible formats.
    if formats == ["all"]:
        channels = get_channels(
            channel_dict,
            FORMATS,
            index,
            extend_aux,
        )
        return channels

    channels = []

    # Include channels from multi_index if 'index' is True.
    if index:
        channels.extend(MULTI_INDEX)

    if formats:
        # If 'formats' is a list, check if all elements are valid.
        for format_ in formats:
            if format_ not in FORMATS + ["all"]:
                raise ValueError(
                    "Invalid format. Please choose from 'per_electron', 'per_pulse',\
                    'per_train', 'all'.",
                )

        # Get the available channels excluding 'pulseId'.
        available_channels = list(channel_dict.keys())
        # raises error if not available, but necessary for pulse_index
        available_channels.remove(PULSE_ALIAS)

        for format_ in formats:
            # Gather channels based on the specified format(s).
            channels.extend(
                key
                for key in available_channels
                if channel_dict[key]["format"] == format_ and key != DLD_AUX_ALIAS
            )
            # Include 'dldAuxChannels' if the format is 'per_pulse' and extend_aux is True.
            # Otherwise, include 'dldAux'.
            if format_ == FORMATS[2] and DLD_AUX_ALIAS in available_channels:
                if extend_aux:
                    channels.extend(
                        channel_dict[DLD_AUX_ALIAS][DLDAUX_CHANNELS].keys(),
                    )
                else:
                    channels.extend([DLD_AUX_ALIAS])

    return channels


def initialize_paths(
    filenames: str | list[str] = None,
    folder: Path = None,
    subfolder: str = "",
    prefix: str = "",
    suffix: str = "",
    extension: str = "parquet",
    paths: list[Path] = None,
) -> list[Path]:
    """
    Initialize the paths for files to be saved/loaded.

    If custom paths are provided, they will be used. Otherwise, paths will be generated based on
    the specified parameters during initialization.

    Args:
        filenames (str | list[str]): The name(s) of the file(s).
        folder (Path): The folder where the files are saved.
        subfolder (str): The subfolder where the files are saved.
        prefix (str): The prefix for the file name.
        suffix (str): The suffix for the file name.
        extension (str): The extension for the file.
        paths (list[Path]): Custom paths for the files.

    Returns:
        list[Path]: The paths for the files.
    """
    # if filenames is string, convert it to a list
    if isinstance(filenames, str):
        filenames = [filenames]

    # Check if the folder and Parquet paths are provided
    if not folder and not paths:
        raise ValueError("Please provide folder or paths.")
    if folder and not filenames:
        raise ValueError("With folder, please provide filenames.")

    # Otherwise create the full path for the Parquet file
    directory = folder.joinpath(subfolder)
    directory.mkdir(parents=True, exist_ok=True)

    if extension:
        extension = f".{extension}"  # if extension is provided, it is prepended with a dot
    if prefix:
        prefix = f"{prefix}_"
    if suffix:
        suffix = f"_{suffix}"
    paths = [directory.joinpath(Path(f"{prefix}{name}{suffix}{extension}")) for name in filenames]

    return paths

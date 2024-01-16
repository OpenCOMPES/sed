from __future__ import annotations

MULTI_INDEX = ["trainId", "pulseId", "electronId"]
DLD_AUX_ALIAS = "dldAux"
DLDAUX_CHANNELS = "dldAuxChannels"
FORMATS = ["per_electron", "per_pulse", "per_train"]


def get_channels(
    channels_dict: dict = None,
    formats: str | list[str] = None,
    index: bool = False,
    extend_aux: bool = False,
    remove_index_from_format: bool = True,
) -> list[str]:
    """
    Returns a list of channels associated with the specified format(s).

    Args:
        channels_dict (dict): The channels dictionary.
        formats (Union[str, List[str]]): The desired format(s)
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
            channels_dict,
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

        # Get the available channels excluding the index
        available_channels = list(channels_dict.keys())
        for ch in MULTI_INDEX:
            if remove_index_from_format and ch in available_channels:
                available_channels.remove(ch)

        for format_ in formats:
            # Gather channels based on the specified format(s).
            channels.extend(
                key
                for key in available_channels
                if channels_dict[key].format == format_ and key != DLD_AUX_ALIAS
            )
            # Include 'dldAuxChannels' if the format is 'per_pulse' and extend_aux is True.
            # Otherwise, include 'dldAux'.
            if format_ == FORMATS[2] and DLD_AUX_ALIAS in available_channels:
                if extend_aux:
                    channels.extend(
                        channels_dict.get(DLD_AUX_ALIAS).dldAuxChannels.keys(),
                    )
                else:
                    channels.extend([DLD_AUX_ALIAS])

    return channels
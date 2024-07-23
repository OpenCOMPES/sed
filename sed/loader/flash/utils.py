from __future__ import annotations


# TODO: move to config
MULTI_INDEX = ["trainId", "pulseId", "electronId"]
PULSE_ALIAS = MULTI_INDEX[1]
FORMATS = ["per_electron", "per_pulse", "per_train"]


def get_channels(
    config_dataframe: dict = {},
    formats: str | list[str] = None,
    index: bool = False,
    extend_aux: bool = False,
) -> list[str]:
    """
    Returns a list of channels associated with the specified format(s).
    'all' returns all channels but 'pulseId' and 'dldAux' (if not extended).

    Args:
        config_dataframe (dict): The config dictionary containing the dataframe keys.
        formats (str | list[str]): The desired format(s)
        ('per_pulse', 'per_electron', 'per_train', 'all').
        index (bool): If True, includes channels from the multiindex.
        extend_aux (bool): If True, includes channels from the subchannels of the auxiliary channel.
                else just includes the auxiliary channel alias.

    Returns:
        List[str]: A list of channels with the specified format(s).
    """
    channel_dict = config_dataframe.get("channels", {})
    aux_alias = config_dataframe.get("aux_alias", "dldAux")

    # If 'formats' is a single string, convert it to a list for uniform processing.
    if isinstance(formats, str):
        formats = [formats]

    # If 'formats' is a string "all", gather all possible formats.
    if formats == ["all"]:
        channels = get_channels(
            config_dataframe,
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
        err_msg = (
            "Invalid format. Please choose from 'per_electron', 'per_pulse', 'per_train', 'all'."
        )
        for format_ in formats:
            if format_ not in FORMATS + ["all"]:
                raise ValueError(err_msg)

        # Get the available channels excluding 'pulseId'.
        available_channels = list(channel_dict.keys())
        # raises error if not available, but necessary for pulse_index
        available_channels.remove(PULSE_ALIAS)

        for format_ in formats:
            # Gather channels based on the specified format(s).
            channels.extend(
                key
                for key in available_channels
                if channel_dict[key]["format"] == format_ and key != aux_alias
            )
            # Include 'dldAuxChannels' if the format is 'per_train' and extend_aux is True.
            # Otherwise, include 'dldAux'.
            if format_ == FORMATS[2] and aux_alias in available_channels:
                if extend_aux:
                    channels.extend(
                        channel_dict[aux_alias]["subChannels"].keys(),
                    )
                else:
                    channels.extend([aux_alias])

    return channels


def get_dtypes(config_dataframe: dict, df_cols: list) -> dict:
    """Returns a dictionary of channels and their corresponding data types.
    Currently Auxiliary channels are not included in the dtype dictionary.

    Args:
        config_dataframe (dict): The config dictionary containing the dataframe keys.
        df_cols (list): A list of channels in the DataFrame.

    Returns:
        dict: A dictionary of channels and their corresponding data types.
    """
    channels_dict = config_dataframe.get("channels", {})
    aux_alias = config_dataframe.get("aux_alias", "dldAux")
    dtypes = {}
    for channel in df_cols:
        try:
            dtypes[channel] = channels_dict[channel].get("dtype")
        except KeyError:
            try:
                dtypes[channel] = channels_dict[aux_alias][channel].get("dtype")
            except KeyError:
                dtypes[channel] = None
    return dtypes

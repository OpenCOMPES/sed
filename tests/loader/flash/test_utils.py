"""Tests for utils functionality"""
from sed.loader.flash.utils import get_channels

# Define expected channels for each format.
ELECTRON_CHANNELS = ["dldPosX", "dldPosY", "dldTimeSteps"]
PULSE_CHANNELS = ["pulserSignAdc", "gmdTunnel"]
TRAIN_CHANNELS = ["timeStamp", "delayStage", "dldAux"]
TRAIN_CHANNELS_EXTENDED = [
    "sampleBias",
    "tofVoltage",
    "extractorVoltage",
    "extractorCurrent",
    "cryoTemperature",
    "sampleTemperature",
    "dldTimeBinSize",
    "timeStamp",
    "delayStage",
]
INDEX_CHANNELS = ["trainId", "pulseId", "electronId"]


def test_get_channels_by_format(config_dataframe: dict) -> None:
    """
    Test function to verify the 'get_channels' method in FlashLoader class for
    retrieving channels based on formats and index inclusion.
    """
    # Initialize the FlashLoader instance with the given config_file.
    ch_dict = config_dataframe

    # Call get_channels method with different format options.

    # Request channels for 'per_electron' format using a list.
    print(ch_dict["channels"])
    format_electron = get_channels(ch_dict, ["per_electron"])

    # Request channels for 'per_pulse' format using a string.
    format_pulse = get_channels(ch_dict, "per_pulse")

    # Request channels for 'per_train' format without expanding the dldAuxChannels.
    format_train = get_channels(ch_dict, "per_train", extend_aux=False)

    # Request channels for 'per_train' format using a list, and expand the dldAuxChannels.
    format_train_extended = get_channels(ch_dict, ["per_train"], extend_aux=True)

    # Request channels for 'all' formats using a list.
    format_all = get_channels(ch_dict, ["all"])

    # Request index channels only.
    format_index = get_channels(ch_dict, index=True)

    # Request 'per_electron' format and include index channels.
    format_index_electron = get_channels(ch_dict, ["per_electron"], index=True)

    # Request 'all' formats and include index channels.
    format_all_index = get_channels(ch_dict, ["all"], index=True)

    # Request 'all' formats and include index channels and extend aux channels
    format_all_index_extend_aux = get_channels(ch_dict, ["all"], index=True, extend_aux=True)

    # Assert that the obtained channels match the expected channels.
    assert set(ELECTRON_CHANNELS) == set(format_electron)
    assert set(TRAIN_CHANNELS_EXTENDED) == set(format_train_extended)
    assert set(TRAIN_CHANNELS) == set(format_train)
    assert set(PULSE_CHANNELS) == set(format_pulse)
    assert set(ELECTRON_CHANNELS + TRAIN_CHANNELS + PULSE_CHANNELS) == set(format_all)
    assert set(INDEX_CHANNELS) == set(format_index)
    assert set(INDEX_CHANNELS + ELECTRON_CHANNELS) == set(format_index_electron)
    assert set(INDEX_CHANNELS + ELECTRON_CHANNELS + TRAIN_CHANNELS + PULSE_CHANNELS) == set(
        format_all_index,
    )
    assert set(
        INDEX_CHANNELS + ELECTRON_CHANNELS + PULSE_CHANNELS + TRAIN_CHANNELS_EXTENDED,
    ) == set(
        format_all_index_extend_aux,
    )

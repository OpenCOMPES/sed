import numpy as np
from pandas import MultiIndex

from sed.loader.fel import MultiIndexCreator


def test_reset_multi_index():
    mi = MultiIndexCreator()
    mi.reset_multi_index()
    assert mi.index_per_electron is None
    assert mi.index_per_pulse is None


def test_create_multi_index_per_electron(pulse_id_array, config_dataframe):
    train_id, np_array = pulse_id_array
    mi = MultiIndexCreator()
    mi.create_multi_index_per_electron(train_id, np_array, 5)

    # Check if the index_per_electron is a MultiIndex and has the correct levels
    assert isinstance(mi.index_per_electron, MultiIndex)
    assert set(mi.index_per_electron.names) == {"trainId", "pulseId", "electronId"}

    # Check if the index_per_electron has the correct number of elements
    array_without_nan = np_array[~np.isnan(np_array)]
    assert len(mi.index_per_electron) == array_without_nan.size

    assert np.all(mi.index_per_electron.get_level_values("trainId").unique() == train_id)
    assert np.all(
        mi.index_per_electron.get_level_values("pulseId").values
        == array_without_nan - config_dataframe["ubid_offset"],
    )

    assert np.all(
        mi.index_per_electron.get_level_values("electronId").values[:5] == [0, 1, 0, 1, 2],
    )

    assert np.all(
        mi.index_per_electron.get_level_values("electronId").values[-5:] == [0, 1, 0, 1, 0],
    )


def test_create_multi_index_per_pulse(gmd_channel_array):
    # can use pulse_id_array as it is also pulse resolved
    train_id, np_array = gmd_channel_array
    mi = MultiIndexCreator()
    mi.create_multi_index_per_pulse(train_id, np_array)

    # Check if the index_per_pulse is a MultiIndex and has the correct levels
    assert isinstance(mi.index_per_pulse, MultiIndex)
    assert set(mi.index_per_pulse.names) == {"trainId", "pulseId"}
    assert len(mi.index_per_pulse) == np_array.size
    print(mi.index_per_pulse.get_level_values("pulseId"))
    assert np.all(mi.index_per_pulse.get_level_values("pulseId").values[:7] == np.arange(0, 7))

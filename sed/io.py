import h5py
import xarray as xr
import numpy as np

def recursive_write_metadata(h5group, node):
    for key, item in node.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes, int, float, list)):
            try:
                h5group.create_dataset(key, data=item)
            except TypeError:
                h5group.create_dataset(key, data=str(item))
                print("saved " + key + " as string")
        elif isinstance(item, dict):
            print(key)
            group = h5group.create_group(key)
            recursive_write_metadata(group, item)
        else:
            try:
                h5group.create_dataset(key, data=str(item))
                print("saved " + key + " as string")
            except:
                raise ValueError('Cannot save %s type'%type(item))

def recursive_parse_metadata(node):
    if isinstance(node, h5py.Group):
        dictionary = {}
        for key, value in node.items():
            dictionary[key] = recursive_parse_metadata(value)

    else:
        dictionary = node[...]
        try:
            dictionary = dictionary.item()
            if isinstance(dictionary, (bytes, bytearray)):
                dictionary = dictionary.decode()
        except ValueError:
            pass

    return dictionary


DEFAULT_UNITS = {
    'X': 'step',
    'Y': 'step',
    't': 'step',
    'tofVoltage': 'V',
    'extractorVoltage': 'V',
    'extractorCurrent': 'A',
    'cryoTemperature': 'K',
    'sampleTemperature': 'K',
    'dldTimeBinSize': 'ns',
    'delay': 'ps',
    'timeStamp': 's',
    'energy': 'eV',
    'kx': '1/A',
    'ky': '1/A'}

def res_to_xarray(res, bin_names, bin_axes, metadata=None):
    """ creates a BinnedArray (xarray subclass) out of the given np.array
    Parameters:
        res: np.array
            nd array of binned data
        bin_names (list): list of names of the binned axes
        bin_axes (list): list of np.arrays with the values of the axes
    Returns:
        ba: BinnedArray (xarray)
            an xarray-like container with binned data, axis, and all available metadata
    """
    dims = bin_names
    coords = {}
    for name, vals in zip(bin_names, bin_axes):
        coords[name] = vals

    xres = xr.DataArray(res, dims=dims, coords=coords)

    for name in bin_names:
        try:
            xres[name].attrs['unit'] = DEFAULT_UNITS[name]
        except KeyError:
            pass

    xres.attrs['units'] = 'counts'
    xres.attrs['long_name'] = 'photoelectron counts'

    if metadata is not None:
        xres.attrs['metadata'] = metadata

    return xres

def xarray_to_h5(data, faddr, mode='w'):
    """ Save xarray formatted data to hdf5
    Args:
        data (xarray.DataArray): input data
        faddr (str): complete file name (including path)
        mode (str): hdf5 read/write mode
    Returns:
    """
    with h5py.File(faddr, mode) as h5File:

        print(f'saving data to {faddr}')

        # Saving data

        ff = h5File.create_group('binned')

        # make a single dataset
        ff.create_dataset('BinnedData', data=data.data)

        # Saving axes
        aa = h5File.create_group("axes")
        ax_n = 0
        for binName in data.dims:
            ds = aa.create_dataset(f'ax{ax_n}', data=data.coords[binName])
            ds.attrs['name'] = binName
            ax_n += 1


        if ('metadata' in data.attrs and isinstance(data.attrs['metadata'], dict)):
            meta_group = h5File.create_group('metadata')

            recursive_write_metadata(meta_group, data.attrs['metadata'])
                   
    print('Saving complete!')



def h5_to_xarray(faddr, mode='r'):
    """ Rear xarray data from formatted hdf5 file
    Args:
        faddr (str): complete file name (including path)
        mode (str): hdf5 read/write mode
    Returns:
        xarray (xarray.DataArray): output xarra data
    """
    with h5py.File(faddr, mode) as h5_file:
        # Reading data array
        try:
            data = h5_file['binned']['BinnedData']
        except KeyError:
            print("Wrong Data Format, data not found")
            raise

        # Reading the axes
        axes = []
        bin_names = []

        try:
            for axis in h5_file['axes']:
                axes.append(h5_file['axes'][axis])
                bin_names.append(h5_file['axes'][axis].attrs['name'])
        except KeyError:
            print("Wrong Data Format, axes not found")
            raise

        # load metadata
        if 'metadata' in h5_file:
            metadata = recursive_parse_metadata(h5_file['metadata'])
            xarray = res_to_xarray(data, bin_names, axes, metadata)

        else:
            xarray = res_to_xarray(data, bin_names, axes)
        return xarray


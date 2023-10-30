Binning with metadata generation, and storing into a NeXus file
===============================================================

In this example, we show how to bin the same data used for example 3,
but using the values for correction/calibration parameters generated in
the example notebook 3, which are locally saved in the file
sed_config.yaml. These data and the corresponding (machine and
processing) metadata are then stored to a NeXus file following the
NXmpes NeXus standard
(https://fairmat-experimental.github.io/nexus-fairmat-proposal/9636feecb79bb32b828b1a9804269573256d7696/classes/contributed_definitions/NXmpes.html#nxmpes)
using the ‘dataconverter’ of the pynxtools package
(https://github.com/FAIRmat-NFDI/pynxtools).

.. code:: ipython3

    %load_ext autoreload
    %autoreload 2
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    import sed
    
    %matplotlib widget

Load Data
---------

.. code:: ipython3

    data_path = '.' # Put in Path to a storage of at least 20 Gbyte free space.
    if not os.path.exists(data_path + "/WSe2.zip"):
        os.system(f"curl -L --output {data_path}/WSe2.zip https://zenodo.org/record/6369728/files/WSe2.zip")
    if not os.path.isdir(data_path + "/Scan049_1") or not os.path.isdir(data_path + "energycal_2019_01_08/"):
        os.system(f"unzip -d {data_path} -o {data_path}/WSe2.zip")

.. code:: ipython3

    metadata = {}
    # manual Meta data. These should ideally come from an Electronic Lab Notebook.
    #General
    metadata['experiment_summary'] = 'WSe2 XUV NIR pump probe data.'
    metadata['entry_title'] = 'Valence Band Dynamics - 800 nm linear s-polarized pump, 0.6 mJ/cm2 absorbed fluence'
    metadata['experiment_title'] = 'Valence band dynamics of 2H-WSe2'
    
    #User
    # Fill general parameters of NXuser
    # TODO: discuss how to deal with multiple users?
    metadata['user0'] = {}
    metadata['user0']['name'] = 'Julian Maklar'
    metadata['user0']['role'] = 'Principal Investigator'
    metadata['user0']['affiliation'] = 'Fritz Haber Institute of the Max Planck Society'
    metadata['user0']['address'] = 'Faradayweg 4-6, 14195 Berlin'
    metadata['user0']['email'] = 'maklar@fhi-berlin.mpg.de'
    
    #NXinstrument
    metadata['instrument'] = {}
    #analyzer
    metadata['instrument']['analyzer']={}
    metadata['instrument']['analyzer']['slow_axes'] = "delay" # the scanned axes
    metadata['instrument']['analyzer']['spatial_resolution'] = 10.
    metadata['instrument']['analyzer']['energy_resolution'] = 110.
    metadata['instrument']['analyzer']['momentum_resolution'] = 0.08
    metadata['instrument']['analyzer']['working_distance'] = 4.
    metadata['instrument']['analyzer']['lens_mode'] = "6kV_kmodem4.0_30VTOF.sav"
    
    #probe beam
    metadata['instrument']['beam']={}
    metadata['instrument']['beam']['probe']={}
    metadata['instrument']['beam']['probe']['incident_energy'] = 21.7
    metadata['instrument']['beam']['probe']['incident_energy_spread'] = 0.11
    metadata['instrument']['beam']['probe']['pulse_duration'] = 20.
    metadata['instrument']['beam']['probe']['frequency'] = 500.
    metadata['instrument']['beam']['probe']['incident_polarization'] = [1, 1, 0, 0] # p pol Stokes vector
    metadata['instrument']['beam']['probe']['extent'] = [80., 80.]
    #pump beam
    metadata['instrument']['beam']['pump']={}
    metadata['instrument']['beam']['pump']['incident_energy'] = 1.55
    metadata['instrument']['beam']['pump']['incident_energy_spread'] = 0.08
    metadata['instrument']['beam']['pump']['pulse_duration'] = 35.
    metadata['instrument']['beam']['pump']['frequency'] = 500.
    metadata['instrument']['beam']['pump']['incident_polarization'] = [1, -1, 0, 0] # s pol Stokes vector
    metadata['instrument']['beam']['pump']['incident_wavelength'] = 800.
    metadata['instrument']['beam']['pump']['average_power'] = 300.
    metadata['instrument']['beam']['pump']['pulse_energy'] = metadata['instrument']['beam']['pump']['average_power']/metadata['instrument']['beam']['pump']['frequency']#µJ
    metadata['instrument']['beam']['pump']['extent'] = [230., 265.]
    metadata['instrument']['beam']['pump']['fluence'] = 0.15
    
    #sample
    metadata['sample']={}
    metadata['sample']['preparation_date'] = '2019-01-13T10:00:00+00:00'
    metadata['sample']['preparation_description'] = 'Cleaved'
    metadata['sample']['sample_history'] = 'Cleaved'
    metadata['sample']['chemical_formula'] = 'WSe2'
    metadata['sample']['description'] = 'Sample'
    metadata['sample']['name'] = 'WSe2 Single Crystal'
    
    metadata['file'] = {}
    metadata['file']["trARPES:Carving:TEMP_RBV"] = 300.
    metadata['file']["trARPES:XGS600:PressureAC:P_RD"] = 5.e-11
    metadata['file']["KTOF:Lens:Extr:I"] = -0.12877
    metadata['file']["KTOF:Lens:UDLD:V"] = 399.99905
    metadata['file']["KTOF:Lens:Sample:V"] = 17.19976
    metadata['file']["KTOF:Apertures:m1.RBV"] = 3.729931
    metadata['file']["KTOF:Apertures:m2.RBV"] = -5.200078
    metadata['file']["KTOF:Apertures:m3.RBV"] = -11.000425
    
    # Sample motor positions
    metadata['file']['trARPES:Carving:TRX.RBV'] = 7.1900000000000004
    metadata['file']['trARPES:Carving:TRY.RBV'] = -6.1700200225439552
    metadata['file']['trARPES:Carving:TRZ.RBV'] = 33.4501953125
    metadata['file']['trARPES:Carving:THT.RBV'] = 423.30500940561586
    metadata['file']['trARPES:Carving:PHI.RBV'] = 0.99931647456264949
    metadata['file']['trARPES:Carving:OMG.RBV'] = 11.002500171914066

.. code:: ipython3

    # The Scan directory
    fdir = data_path + '/Scan049_1'
    # create sed processor using the config file, and collect the meta data from the files:
    sp = sed.SedProcessor(folder=fdir, config="../sed/config/mpes_example_config.yaml", metadata=metadata, collect_metadata=True)

.. code:: ipython3

    # Apply jittering to X, Y, t, ADC columns.
    sp.add_jitter()

.. code:: ipython3

    # Calculate machine-coordinate data for pose adjustment
    sp.bin_and_load_momentum_calibration(df_partitions=10, plane=33, width=10, apply=True)

.. code:: ipython3

    # Adjust pose alignment, using stored distortion correction
    sp.pose_adjustment(xtrans=8, ytrans=7, angle=-4, apply=True, use_correction=True)

.. code:: ipython3

    # Apply stored momentum correction
    sp.apply_momentum_correction()

.. code:: ipython3

    # Apply stored config momentum calibration
    sp.apply_momentum_calibration()

.. code:: ipython3

    # Apply stored config energy correction
    sp.apply_energy_correction()

.. code:: ipython3

    # Apply stored config energy calibration
    sp.append_energy_axis()

.. code:: ipython3

    # Apply delay calibration
    delay_range = (-500, 1500)
    sp.calibrate_delay_axis(delay_range=delay_range, preview=True)

Compute final data volume
-------------------------

.. code:: ipython3

    axes = ['kx', 'ky', 'energy', 'delay']
    bins = [100, 100, 200, 50]
    ranges = [[-2, 2], [-2, 2], [-4, 2], [-600, 1600]]
    res = sp.compute(bins=bins, axes=axes, ranges=ranges)

.. code:: ipython3

    # save to NXmpes NeXus (including standardized metadata)
    sp.save(data_path + "/binned.nxs")

.. code:: ipython3

    # Visualization (requires JupyterLab)
    from jupyterlab_h5web import H5Web
    H5Web(data_path + "/binned.nxs")


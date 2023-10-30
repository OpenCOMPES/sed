.. code:: ipython3

    %load_ext autoreload
    %autoreload 2
    
    from sed import SedProcessor
    import sed
    import numpy as np
    
    # %matplotlib inline
    %matplotlib widget
    import matplotlib.pyplot as plt

Try to calibrate energy
=======================

Spin-integrated branch, E_TOF=10eV
----------------------------------

single scan, move sample bias manually every 2000 pulses.

.. code:: ipython3

    sp = SedProcessor(runs=[44638], config="config_flash_energy_calib.yaml", system_config={})

.. code:: ipython3

    sp.add_jitter()

.. code:: ipython3

    axes = ['sampleBias', 'dldTime']
    bins = [6, 500]
    ranges = [[0,6], [40000, 55000]]
    res = sp.compute(bins=bins, axes=axes, ranges=ranges)

.. code:: ipython3

    sp.load_bias_series(binned_data=res)

.. code:: ipython3

    ranges=(44500, 46000)
    ref_id=3
    sp.find_bias_peaks(ranges=ranges, ref_id=ref_id)

.. code:: ipython3

    ref_id=3
    ref_energy=-.3
    sp.calibrate_energy_axis(ref_id=ref_id, ref_energy=ref_energy, method="lstsq", order=3)

.. code:: ipython3

    ref_id=3
    ref_energy=-.3
    sp.calibrate_energy_axis(ref_id=ref_id, ref_energy=ref_energy, method="lmfit")

.. code:: ipython3

    sp.append_energy_axis(preview=True)

.. code:: ipython3

    axes = ['sampleBias', 'energy']
    bins = [6, 1000]
    ranges = [[0,6], [-5, 5]]
    res = sp.compute(bins=bins, axes=axes, ranges=ranges)

.. code:: ipython3

    plt.figure()
    res[3,:].plot()

.. code:: ipython3

    axes = ['sampleBias', 'energy', 'dldPosX']
    bins = [6, 100, 480]
    ranges = [[0,6], [-2, 1], [420,900]]
    res = sp.compute(bins=bins, axes=axes, ranges=ranges)

.. code:: ipython3

    plt.figure()
    res[3, :, :].plot()


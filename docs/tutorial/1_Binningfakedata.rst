Binning demonstration on locally generated fake data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, we generate a table with random data simulating a
single event dataset. We showcase the binning method, first on a simple
single table using the bin_partition method and then in the distributed
mehthod bin_dataframe, using daks dataframes. The first method is never
really called directly, as it is simply the function called by the
bin_dataframe on each partition of the dask dataframe.

.. code:: ipython3

    import sys
    
    import dask
    import numpy as np
    import pandas as pd
    import dask.dataframe
    
    import matplotlib.pyplot as plt
    
    sys.path.append("../")
    from sed.binning import bin_partition, bin_dataframe

Generate Fake Data
------------------

.. code:: ipython3

    n_pts = 100000
    cols = ["posx", "posy", "energy"]
    df = pd.DataFrame(np.random.randn(n_pts, len(cols)), columns=cols)
    df




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>posx</th>
          <th>posy</th>
          <th>energy</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.920564</td>
          <td>-1.068583</td>
          <td>1.011868</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.101308</td>
          <td>-1.133177</td>
          <td>2.264009</td>
        </tr>
        <tr>
          <th>2</th>
          <td>-2.175991</td>
          <td>0.469750</td>
          <td>-0.366066</td>
        </tr>
        <tr>
          <th>3</th>
          <td>-1.414038</td>
          <td>1.505585</td>
          <td>-1.168827</td>
        </tr>
        <tr>
          <th>4</th>
          <td>-1.239659</td>
          <td>-0.401433</td>
          <td>0.055166</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>-0.386748</td>
          <td>0.796456</td>
          <td>1.208073</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>-1.012029</td>
          <td>0.886339</td>
          <td>-0.616620</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.486870</td>
          <td>-0.143890</td>
          <td>-1.772194</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>-1.169187</td>
          <td>2.560502</td>
          <td>-0.039270</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>-0.813120</td>
          <td>1.152984</td>
          <td>0.006634</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 3 columns</p>
    </div>



Define the binning range
------------------------

.. code:: ipython3

    binAxes = ["posx", "posy", "energy"]
    nBins = [120, 120, 120]
    binRanges = [(-2, 2), (-2, 2), (-2, 2)]
    coords = {ax: np.linspace(r[0], r[1], n) for ax, r, n in zip(binAxes, binRanges, nBins)}

Compute the binning along the pandas dataframe
----------------------------------------------

.. code:: ipython3

    %%time
    res = bin_partition(
        part=df,
        bins=nBins,
        axes=binAxes,
        ranges=binRanges,
        hist_mode="numba",
    )


.. parsed-literal::

    CPU times: user 1.13 s, sys: 18.5 ms, total: 1.14 s
    Wall time: 1.19 s


.. code:: ipython3

    fig, axs = plt.subplots(1, 3, figsize=(8, 2.5), constrained_layout=True)
    for i in range(3):
        axs[i].imshow(res.sum(i))



.. image:: 1_Binningfakedata_files/1_Binningfakedata_8_0.png


Transform to dask dataframe
---------------------------

.. code:: ipython3

    ddf = dask.dataframe.from_pandas(df, npartitions=50)
    ddf




.. raw:: html

    <div><strong>Dask DataFrame Structure:</strong></div>
    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>posx</th>
          <th>posy</th>
          <th>energy</th>
        </tr>
        <tr>
          <th>npartitions=50</th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>float64</td>
          <td>float64</td>
          <td>float64</td>
        </tr>
        <tr>
          <th>2000</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>98000</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
      </tbody>
    </table>
    </div>
    <div>Dask Name: from_pandas, 1 graph layer</div>



compute distributed binning on the partitioned dask dataframe
-------------------------------------------------------------

In this example, the small dataset does not give significant improvement
over the pandas implementation, at least using this number of
partitions. A single partition would be faster (you can try…) but we use
multiple for demonstration purpouses.

.. code:: ipython3

    %%time
    res = bin_dataframe(
        df=ddf,
        bins=nBins,
        axes=binAxes,
        ranges=binRanges,
        hist_mode="numba",
    )



.. parsed-literal::

      0%|          | 0/50 [00:00<?, ?it/s]


.. parsed-literal::

    CPU times: user 504 ms, sys: 280 ms, total: 784 ms
    Wall time: 699 ms


.. code:: ipython3

    fig, axs = plt.subplots(1, 3, figsize=(8, 2.5), constrained_layout=True)
    for dim, ax in zip(binAxes, axs):
        res.sum(dim).plot(ax=ax)



.. image:: 1_Binningfakedata_files/1_Binningfakedata_13_0.png



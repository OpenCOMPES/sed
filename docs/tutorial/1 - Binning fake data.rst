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
          <td>-0.424826</td>
          <td>-0.502427</td>
          <td>-0.596955</td>
        </tr>
        <tr>
          <th>1</th>
          <td>-0.323580</td>
          <td>-1.945639</td>
          <td>-0.329882</td>
        </tr>
        <tr>
          <th>2</th>
          <td>-0.199051</td>
          <td>-0.678583</td>
          <td>-0.286432</td>
        </tr>
        <tr>
          <th>3</th>
          <td>-0.458054</td>
          <td>-0.378451</td>
          <td>-0.161591</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.028276</td>
          <td>1.780522</td>
          <td>0.861112</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.648457</td>
          <td>1.017227</td>
          <td>0.681250</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.358100</td>
          <td>-3.356823</td>
          <td>-0.040907</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>0.771047</td>
          <td>0.146662</td>
          <td>-0.399212</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>-0.011129</td>
          <td>0.333914</td>
          <td>1.523470</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>0.804708</td>
          <td>-0.263338</td>
          <td>-0.038946</td>
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

    CPU times: user 1.54 s, sys: 35 ms, total: 1.58 s
    Wall time: 1.62 s


.. code:: ipython3

    fig, axs = plt.subplots(1, 3, figsize=(8, 2.5), constrained_layout=True)
    for i in range(3):
        axs[i].imshow(res.sum(i))



.. image:: 1%20-%20Binning%20fake%20data_files/1%20-%20Binning%20fake%20data_8_0.png


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

    CPU times: user 686 ms, sys: 321 ms, total: 1.01 s
    Wall time: 946 ms


.. code:: ipython3

    fig, axs = plt.subplots(1, 3, figsize=(8, 2.5), constrained_layout=True)
    for dim, ax in zip(binAxes, axs):
        res.sum(dim).plot(ax=ax)



.. image:: 1%20-%20Binning%20fake%20data_files/1%20-%20Binning%20fake%20data_13_0.png



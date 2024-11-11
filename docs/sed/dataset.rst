Dataset
===================================================

SED comes with the ability to download and extract any URL based
datasets. By default, user can the “WSe2”, “TaS2” and “Gd_W110” datasets
but easy to extend this list.

Getting datasets
------------------------

.. code:: python

   import os
   from sed.dataset import dataset

get()
^^^^^

The “get” just needs the data name, but another root_dir can be provided.

Try to interrupt the download process and restart to see that it continues the download from where it stopped

.. code:: python

   dataset.get("WSe2", remove_zip = False)

.. parsed-literal::

   Using default data path for "WSe2": "<user_path>/datasets/WSe2"

   3%|▎         | 152M/5.73G [00:02<01:24, 71.3MB/s]

   Using default data path for "WSe2": "<user_path>/datasets/WSe2"

   100%|██████████| 5.73G/5.73G [01:09<00:00, 54.3MB/s]

   Download complete.

Not providing “remove_zip” at all will by default delete the zip file after extraction

.. code:: python

   dataset.get("WSe2")

Setting the “use_existing” keyword to False allows to download the data in another location. Default is to use existing data

.. code:: python

   dataset.get("WSe2", root_dir = "new_datasets", use_existing=False)

.. parsed-literal::

   Using specified data path for "WSe2": "<user_path>/new_datasets/datasets/WSe2"
   Created new directory at <user_path>/new_datasets/datasets/WSe2


     3%|▎         | 152M/5.73G [00:02<01:24, 71.3MB/s]


Interrupting extraction has similar behavior to download and just continues from where it stopped.

Or if user deletes the extracted documents, it re-extracts from zip file

.. code:: python

   dataset.get("WSe2", remove_zip = False)

   ## Try to remove some files and rerun this command.

.. parsed-literal::

   Using default data path for "WSe2": "<user_path>/datasets/WSe2"
   WSe2 data is already fully downloaded.


   5.73GB [00:00, 12.6MB/s]

   Download complete.
   Extracting WSe2 data...



   100%|██████████| 113/113 [02:41<00:00,  1.43s/file]

   WSe2 data extracted successfully.

remove()
^^^^^^^^

“remove” allows removal of some or all instances of existing data

This would remove only one of the two existing paths

.. code:: python

   dataset.remove("WSe2", instance = dataset.existing_data_paths[0])

.. parsed-literal::

   Removed <user_path>/datasets/WSe2

This removes all instances, if any present

.. code:: python

   dataset.remove("WSe2")

.. parsed-literal::

   WSe2 data is not present.

Attributes useful for user
^^^^^^^^^^^^^^^^^^^^^^^^^^

All available datasets after looking at module, user and folder levels

.. code:: python

   dataset.available

.. parsed-literal::

   ['WSe2', 'TaS2', 'Gd_W110']

The dir and subdirs where data is located

.. code:: python

   dataset.dir

.. parsed-literal::

   '<user_path>/datasets/WSe2'

.. code:: python

   dataset.subdirs

.. parsed-literal::

   ['<user_path>/datasets/WSe2/Scan049_1',
    '<user_path>/datasets/WSe2/energycal_2019_01_08']

Existing locations where data is present

.. code:: python

   dataset.existing_data_paths

.. parsed-literal::

   ['<user_path>/new_dataset/datasets/WSe2',
    '<user_path>/datasets/WSe2']

Example of adding custom datasets
---------------------------------

DatasetsManager
^^^^^^^^^^^^^^^

Allows to add or remove datasets in json file at any level (module, user, folder).

Looks at all levels to give the available datasets

.. code:: python

   import os
   from sed.dataset import DatasetsManager

We add a new dataset to both folder and user levels

This dataset also has “rearrange_files” set to True, which takes all files in subfolders and puts them in the main dataset specific directory

.. code:: python

   example_dset_name = "Example"
   example_dset_info = {}

   example_dset_info["url"] = "https://example-dataset.com/download" # not a real path
   example_dset_info["subdirs"] = ["Example_subdir"]
   example_dset_info["rearrange_files"] = True

   DatasetsManager.add(data_name=example_dset_name, info=example_dset_info, levels=["folder", "user"])

.. parsed-literal::

   Added Example dataset to folder datasets.json
   Added Example dataset to user datasets.json

datasets.json should be available in execution folder after this

.. code:: python

   assert os.path.exists("./datasets.json")
   dataset.available

.. parsed-literal::

   ['Example', 'WSe2', 'TaS2', 'Gd_W110']

This will remove the Example dataset from the user json file

.. code:: python

   DatasetsManager.remove(data_name=example_dset_name, levels=["user"])

.. parsed-literal::

   Removed Example dataset from user datasets.json

Adding dataset that already exists will give an error. Likewise, removing one that doesn’t exist

.. code:: python

   # This should give an error
   DatasetsManager.add(data_name=example_dset_name, info=example_dset_info, levels=["folder"])

.. parsed-literal::

   ValueError: Dataset Example already exists in folder datasets.json.


Now that dataset.json with Example exists in current dir, lets try to fetch it

.. code:: python

   dataset.get("Example")

.. parsed-literal::

   Using default data path for "Example": "<user_path>/datasets/Example"
   Created new directory at <user_path>/datasets/Example
   Download complete.
   Extracting Example data...


   100%|██████████| 4/4 [00:00<00:00, 28.10file/s]

   Example data extracted successfully.
   Removed Example.zip file.
   Rearranging files in Example_subdir.



   100%|██████████| 3/3 [00:00<00:00, 696.11file/s]

   File movement complete.
   Rearranging complete.

.. code:: python

   print(dataset.dir)
   print(dataset.subdirs)

.. parsed-literal::

   <user_path>/datasets/Example
   []

lets download to another location

.. code:: python

   dataset.get("Example", root_dir = "new_datasets", use_existing = False)

.. parsed-literal.. parsed-literal::

   Using specified data path for "Example": "<user_path>/new_datasets/datasets/Example"
   Created new directory at <user_path>/new_datasets/datasets/Example
   Download complete.
   Extracting Example data...


   100%|██████████| 4/4 [00:00<00:00, 28.28file/s]

   Example data extracted successfully.
   Removed Example.zip file.
   Rearranging files in Example_subdir.



   100%|██████████| 3/3 [00:00<00:00, 546.16file/s]

   File movement complete.
   Rearranging complete.

we can remove one instance

.. code:: python

   print(dataset.existing_data_paths)
   path_to_remove = dataset.existing_data_paths[0]

.. parsed-literal::

   ['<user_path>/new_datasets/datasets/Example', '<user_path>/datasets/Example']

.. code:: python

   dataset.remove(data_name="Example", instance=path_to_remove)

.. parsed-literal::

   Removed <user_path>/new_datasets/datasets/Example

.. code:: python

   assert not os.path.exists(path_to_remove)

.. code:: python

   print(dataset.existing_data_paths)

.. parsed-literal::

   ['<user_path>/datasets/Example']

Default datasets.json
---------------------

.. literalinclude::  ../../sed/dataset/datasets.json
   :language: json

API
------------------------
.. automodule::  sed.dataset.dataset
   :members:
   :undoc-members:

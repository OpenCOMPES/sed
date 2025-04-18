# Dataset

## Overview
SED comes with the ability to download and extract any URL-based datasets. By default, users can use the datasets:
- `WSe2`
- `TaS2`
- `Gd_W110`
- `W110`

It is easy to extend this list using a JSON file.

---

## Getting Datasets

### Importing Required Modules
```python
import os
from sed.dataset import dataset
```

### `get()` Method
The `get` method requires only the dataset name, but an alternative `root_dir` can be provided.

Try interrupting the download process and restarting it to see that it resumes from where it stopped.

```python
dataset.get("WSe2", remove_zip=False)
```

Example Output:
```
Using default data path for "WSe2": "<user_path>/datasets/WSe2"

3%|▎         | 152M/5.73G [00:02<01:24, 71.3MB/s]

Using default data path for "WSe2": "<user_path>/datasets/WSe2"

100%|██████████| 5.73G/5.73G [01:09<00:00, 54.3MB/s]

Download complete.
```

By default, not providing `remove_zip=False` will delete the zip file after extraction:
```python
dataset.get("WSe2")
```

Setting `use_existing=False` allows downloading the data to a new location instead of using existing data.
```python
dataset.get("WSe2", root_dir="new_datasets", use_existing=False)
```

Example Output:
```
Using specified data path for "WSe2": "<user_path>/new_datasets/datasets/WSe2"
Created new directory at <user_path>/new_datasets/datasets/WSe2
```

Interrupting extraction here behaves similarly, resuming from where it stopped.

If the extracted files are deleted, rerunning this command below will re-extract from the zip file:
```python
dataset.get("WSe2", remove_zip=False)
```

Example Output:
```
Using default data path for "WSe2": "<user_path>/datasets/WSe2"
WSe2 data is already fully downloaded.

5.73GB [00:00, 12.6MB/s]

Download complete.
Extracting WSe2 data...

100%|██████████| 113/113 [02:41<00:00, 1.43s/file]

WSe2 data extracted successfully.
```
and this does not delete the zip file.

---

## `remove()` Method

The `remove` method allows removing some or all instances of existing data.

Remove only one instance:
```python
dataset.remove("WSe2", instance=dataset.existing_data_paths[0])
```

Example Output:
```
Removed <user_path>/datasets/WSe2
```

Remove all instances:
```python
dataset.remove("WSe2")
```

Example Output:
```
WSe2 data is not present.
```

---

## Useful Attributes

### Available Datasets
```python
dataset.available
```

Example Output:
```
['WSe2', 'TaS2', 'Gd_W110', 'W110']
```

### Data Directory
```python
dataset.dir
```

Example Output:
```
'<user_path>/datasets/WSe2'
```

### Subdirectories
```python
dataset.subdirs
```

Example Output:
```
['<user_path>/datasets/WSe2/Scan049_1',
 '<user_path>/datasets/WSe2/energycal_2019_01_08']
```

### Existing Data Paths
```python
dataset.existing_data_paths
```

Example Output:
```
['<user_path>/new_dataset/datasets/WSe2',
 '<user_path>/datasets/WSe2']
```

---

## Example: Adding Custom Datasets

### `DatasetsManager`
Allows adding or removing datasets in a JSON file at different levels (module, user, folder). It also checks all levels to list available datasets.

```python
import os
from sed.dataset import DatasetsManager
```

#### Adding a New Dataset
This example adds a dataset to both the folder and user levels. Setting `rearrange_files=True` moves all files from subfolders into the main dataset directory.

```python
example_dset_name = "Example"
example_dset_info = {
    "url": "https://example-dataset.com/download",  # Not a real path
    "subdirs": ["Example_subdir"],
    "rearrange_files": True
}

DatasetsManager.add(data_name=example_dset_name, info=example_dset_info, levels=["folder", "user"])
```

Example Output:
```
Added Example dataset to folder datasets.json
Added Example dataset to user datasets.json
```

Verify that `datasets.json` is created:
```python
assert os.path.exists("./datasets.json")
dataset.available
```

Example Output:
```
['Example', 'WSe2', 'TaS2', 'Gd_W110']
```

#### Removing a Dataset
Remove the Example dataset from the user JSON file:
```python
DatasetsManager.remove(data_name=example_dset_name, levels=["user"])
```

Example Output:
```
Removed Example dataset from user datasets.json
```

Adding an already existing dataset will result in an error:
```python
DatasetsManager.add(data_name=example_dset_name, info=example_dset_info, levels=["folder"])
```

Example Output:
```
ValueError: Dataset Example already exists in folder datasets.json.
```

#### Downloading the Example Dataset
```python
dataset.get("Example")
```

Example Output:
```
Using default data path for "Example": "<user_path>/datasets/Example"
Created new directory at <user_path>/datasets/Example
Download complete.
Extracting Example data...

100%|██████████| 4/4 [00:00<00:00, 28.10file/s]

Example data extracted successfully.
```

#### Download to Another Location
```python
dataset.get("Example", root_dir="new_datasets", use_existing=False)
```

Example Output:
```
Using specified data path for "Example": "<user_path>/new_datasets/datasets/Example"
Created new directory at <user_path>/new_datasets/datasets/Example
```

#### Removing an Instance
```python
print(dataset.existing_data_paths)
path_to_remove = dataset.existing_data_paths[0]
dataset.remove(data_name="Example", instance=path_to_remove)
```

Example Output:
```
Removed <user_path>/new_datasets/datasets/Example
```

Verify that the path was removed:
```python
assert not os.path.exists(path_to_remove)
```

```python
print(dataset.existing_data_paths)
```

Example Output:
```
['<user_path>/datasets/Example']
```

---

## Default datasets.json

```{literalinclude} ../../src/sed/config/datasets.json
:language: json
```

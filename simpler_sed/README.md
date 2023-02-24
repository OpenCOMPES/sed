[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
# [WIP] SXP Data Analysis Toolbox
A simplified version of [SED](https://github.com/OpenCOMPES/sed), [HEXTOF](https://github.com/momentoscope/hextof-processor) and [MPES](https://github.com/mpes-kit/mpes) libraries to be used when analyzing photoemision spectroscopy data at the [SXP instrument](https://www.xfel.eu/facility/instruments/sxp/instrument/index_eng.html) of [European XFEL](https://www.xfel.eu/).

## Known limitations
So far, this library is especifically tailored to cope with data measured at FLASH using the HEXTOF (high energy X-ray time of flight) instrument.

## How to build a kernel to be used in Maxwell


- Create a virtual environment in a directory you have write access to it, e.g., your user's home folder.

```
python3 -m venv kernel_name
```

- Source the venv to activate it

```
source ./kernel_name/bin/activate
```

- To avoid the use of the `pip` module from the system, explicitly append the path where your venv is located.

- Upgrade `pip`

```
./kernel_name/bin/python -m pip install -U pip
```

- Install `ipykernel`

```
./kernel_name/bin/python -m pip install ipykernel
```

- Create the kernel to be used in Maxwell in the `user` namespace and give it proper name

```
./kernel_name/bin/python -m ipykernel install --user --name=kernel_name
```

- Navigate to the folder where you clone this repository and `pip install` it in editable mode, in case you want to modify it.

```
pip install -e .
```

Now, the kernel you just created should be listed in the list of available kernels in Maxwell.

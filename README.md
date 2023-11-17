
[![Documentation Status](https://github.com/OpenCOMPES/sed/actions/workflows/documentation.yml/badge.svg)](https://opencompes.github.io/sed/)
![](https://github.com/OpenCOMPES/sed/actions/workflows/linting.yml/badge.svg?branch=main)
![](https://github.com/OpenCOMPES/sed/actions/workflows/testing_multiversion.yml/badge.svg?branch=main)
![](https://img.shields.io/pypi/pyversions/sed-processor)
![](https://img.shields.io/pypi/l/sed-processor)
![](https://img.shields.io/pypi/v/sed-processor)
[![Coverage Status](https://coveralls.io/repos/github/OpenCOMPES/sed/badge.svg?branch=main&kill_cache=1)](https://coveralls.io/github/OpenCOMPES/sed?branch=main)


Single Event Data Frame Processor: Backend to handle photoelectron resolved datastreams

# Installation

## Pip (for users)

- Create a new virtual environment using either venv, pyenv, conda etc. See below for example.

- Install sed, distributed as sed-processor on PyPI

```
pip install sed-processor
```
- This should install all the requirements to run `sed` in your environment.

- If you intend to work with jupyter notebooks, it is helpfull to install a jupyter kernel of your environment. This can be done, once activating your environment, by typing:

```
python -m ipykernel install --user --name=sed_kernel
```

- To create an environment using venv, use

```
python -m venv .sed-venv
```

- To activate your environment:

```
source .sed-venv/bin/activate
```



## Poetry (for developers)

- Prerequisites:
  + poetry: https://python-poetry.org/docs/

- Create a virtual environment by typing:

```python
poetry shell
```

- A new shell will be spawn with the new environment activated


- Install the dependencies from the `pyproject.toml` by typing:

```python
poetry install
```

- If you wish to use the virtual environment created by poetry to work in a Jupyter notebook, you first need to install the optional notebook dependencies and then create a Jupyter kernel for that.

  + Install the optional dependencies ipykernel and jupyter

  ```python
  poetry install -E notebook
  ```

  + Make sure to run the command below within your virtual environment ('poetry run' ensures this) by typing:

  ```python
  poetry run ipython kernel install --user --name=sed_poetry
  ```

  + The new kernel will be eligible now from your kernels list in Jupyter

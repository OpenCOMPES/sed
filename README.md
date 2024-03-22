[![Documentation Status](https://github.com/OpenCOMPES/sed/actions/workflows/documentation.yml/badge.svg)](https://opencompes.github.io/sed/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![](https://github.com/OpenCOMPES/sed/actions/workflows/linting.yml/badge.svg?branch=main)
![](https://github.com/OpenCOMPES/sed/actions/workflows/testing_multiversion.yml/badge.svg?branch=main)
![](https://img.shields.io/pypi/pyversions/sed-processor)
![](https://img.shields.io/pypi/l/sed-processor)
[![](https://img.shields.io/pypi/v/sed-processor)](https://pypi.org/project/sed-processor)
[![Coverage Status](https://coveralls.io/repos/github/OpenCOMPES/sed/badge.svg?branch=main&kill_cache=1)](https://coveralls.io/github/OpenCOMPES/sed?branch=main)

Backend to handle photoelectron resolved datastreams.

# Installation

## Pip (for users)

- Create a new virtual environment using either venv, pyenv, conda, etc. See below for an example.

```bash
python -m venv .sed-venv
```

- Activate your environment:

```bash
source .sed-venv/bin/activate
```

- Install `sed`, distributed as `sed-processor` on PyPI:

```bash
pip install sed-processor
```

- This should install all the requirements to run `sed` in your environment.

- If you intend to work with Jupyter notebooks, it is helpful to install a Jupyter kernel for your environment. This can be done, once your environment is activated, by typing:

```bash
python -m ipykernel install --user --name=sed_kernel
```
## For Contributors

To contribute to the development of `sed`, you can follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/OpenCOMPES/sed.git
cd sed
```

2. Install the repository in editable mode:

```bash
pip install -e .
```

Now you have the development version of `sed` installed in your local environment. Feel free to make changes and submit pull requests.

## Poetry (for maintainers)

- Prerequisites:
  + Poetry: https://python-poetry.org/docs/

- Create a virtual environment by typing:

```bash
poetry shell
```

- A new shell will be spawned with the new environment activated.

- Install the dependencies from the `pyproject.toml` by typing:

```bash
poetry install --with dev, docs
```

- If you wish to use the virtual environment created by Poetry to work in a Jupyter notebook, you first need to install the optional notebook dependencies and then create a Jupyter kernel for that.

  + Install the optional dependencies `ipykernel` and `jupyter`:

  ```bash
  poetry install -E notebook
  ```

  + Make sure to run the command below within your virtual environment (`poetry run` ensures this) by typing:

  ```bash
  poetry run ipython kernel install --user --name=sed_poetry
  ```

  + The new kernel will now be available in your Jupyter kernels list.

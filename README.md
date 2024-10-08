[![Documentation Status](https://github.com/OpenCOMPES/sed/actions/workflows/documentation.yml/badge.svg)](https://opencompes.github.io/sed/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![](https://github.com/OpenCOMPES/sed/actions/workflows/linting.yml/badge.svg?branch=main)
![](https://github.com/OpenCOMPES/sed/actions/workflows/testing_multiversion.yml/badge.svg?branch=main)
![](https://img.shields.io/pypi/pyversions/sed-processor)
![](https://img.shields.io/pypi/l/sed-processor)
[![](https://img.shields.io/pypi/v/sed-processor)](https://pypi.org/project/sed-processor)
[![Coverage Status](https://coveralls.io/repos/github/OpenCOMPES/sed/badge.svg?branch=main&kill_cache=1)](https://coveralls.io/github/OpenCOMPES/sed?branch=main)

Backend to handle photoelectron resolved datastreams.

# Table of Contents
[Installation](#installation)
  - [For Users (pip)](#for-users-pip)
  - [For Contributors (pip)](#for-contributors-pip)
  - [For Maintainers (poetry)](#for-maintainers-poetry)

# Installation

## For Users (pip)

### Prerequisites
- Python 3.8+
- pip

### Steps
- Create a new virtual environment using either venv, pyenv, conda, etc. See below for an example.

```bash
python -m venv .sed-venv
```

- Activate your environment:

```bash
# On macOS/Linux
source .sed-venv/bin/activate

# On Windows
.sed-venv\Scripts\activate
```

- Install `sed`, distributed as `sed-processor` on PyPI:

```bash
pip install sed-processor[all]
```

- If you intend to work with Jupyter notebooks, it is helpful to install a Jupyter kernel for your environment. This can be done, once your environment is activated, by typing:

```bash
python -m ipykernel install --user --name=sed_kernel
```

- If you do not use Jupyter Notebook or Jupyter Lab, you can skip the installing those dependencies

```bash
pip install sed-processor
```

## For Contributors (pip)

### Prerequisites
- Git
- Python 3.8+
- pip

### Steps
1. Clone the repository:

```bash
git clone https://github.com/OpenCOMPES/sed.git
cd sed
```

2. Create and activate a virtual environment:

```bash
# Create a virtual environment
python -m venv .sed-dev

# Activate the virtual environment
# On macOS/Linux
source .sed-dev/bin/activate

# On Windows
.sed-dev\Scripts\activate
```

3. Install the repository in editable mode with all dependencies:

```bash
pip install -e .[all]
```

Now you have the development version of `sed` installed in your local environment. Feel free to make changes and submit pull requests.

## For Maintainers (poetry)

### Prerequisites
- Poetry: [Poetry Installation](https://python-poetry.org/docs/#installation)

### Steps
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

  - Install the optional dependencies:

  ```bash
  poetry install -E notebook
  ```

  - Make sure to run the command below within your virtual environment (`poetry run` ensures this) by typing:

  ```bash
  poetry run ipython kernel install --user --name=sed_poetry
  ```

  - The new kernel will now be available in your Jupyter kernels list.

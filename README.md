# sed
[![Documentation Status](https://readthedocs.org/projects/sed/badge/?version=latest)](https://sed.readthedocs.io/en/latest/?badge=latest)


Single Event Data Frame Processor: Backend to handle photoelectron resolved datastreams

# Installation

## Conda approach

Clone this repository and cd to its root folder.
Create a new environment by typing:
```
conda env create -f env.yml
```
This should install all the requirements to run `sed` in your environment.
To activate your environment:
```
conda activate sed_conda
```
If you intend to work with jupyter notebooks, it is helpfull to install a jupyter kernel of your environment. This can be done, once activating your environment, by typing:
```
python -m ipykernel install --user --name=sed_conda
```


## Poetry approach (better, but more complex)

- Prerequisites:
  + poetry: https://python-poetry.org/docs/
  + pyenv: https://github.com/pyenv/pyenv

- Clone this repository and check the python version within the `[tool.poetry.dependencies]` section of the `pyproject.toml` file
  + If your system is using a different Python version, use `pyenv` to create and activate a Python version compatible with the specifications from the `pyproject.toml`. See [pyenv basic usage](https://github.com/pyenv/pyenv)
- Create a virtual environment by typing:
```python
poetry shell
```
  + A new shell will be spawn with the new environment activated

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

# sed
Single Event Data Frame Processor: Backend to handle photoelectron resolved datastreams

# Installation

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

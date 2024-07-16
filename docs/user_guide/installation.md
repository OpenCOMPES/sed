# Installation

```{attention}
Requires Python 3.9+ and pip installed.
```

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

- If you do not use Jupyter Notebook or Jupyter Lab, you can skip the installing those dependencies:

```bash
pip install sed-processor
```

```{note}
If you intend to work with Jupyter notebooks, it is helpful to install a Jupyter kernel for your environment. This can be done, once your environment is activated, by typing:
```bash
python -m ipykernel install --user --name=sed_kernel
```

# Development version

```{attention}
Requires Git, Python 3.9+ and pip installed.
```

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

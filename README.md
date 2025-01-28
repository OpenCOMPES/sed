[![Documentation Status](https://github.com/OpenCOMPES/sed/actions/workflows/documentation.yml/badge.svg)](https://opencompes.github.io/docs/sed/stable/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![](https://github.com/OpenCOMPES/sed/actions/workflows/linting.yml/badge.svg?branch=main)
![](https://github.com/OpenCOMPES/sed/actions/workflows/testing_multiversion.yml/badge.svg?branch=main)
![](https://img.shields.io/pypi/pyversions/sed-processor)
![](https://img.shields.io/pypi/l/sed-processor)
[![](https://img.shields.io/pypi/v/sed-processor)](https://pypi.org/project/sed-processor)
[![Coverage Status](https://coveralls.io/repos/github/OpenCOMPES/sed/badge.svg?branch=main&kill_cache=1)](https://coveralls.io/github/OpenCOMPES/sed?branch=main)

**sed-processor** is a backend to process and bin multidimensional single-event datastreams, with the intended primary use case in multidimensional photoelectron spectroscopy using time-of-flight instruments.

It builds on [Dask](https://www.dask.org/) dataframes, where each column represents a multidimensional "coordinate" such as position, time-of-flight, pump-probe delay etc., and each entry represents one electron. The `SedProcessor` class provides a single user entry point, and provides functions for handling various workflows for coordinate transformation, e.g. corrections and calibrations.

Furthermore, "sed-processor" provides fast and parallelized binning routines to compute multidimensional histograms from the processed dataframes in a delayed fashion, thus reducing requirements on cpu power and memory consumption.

Finally, in contains several export routines, including export into the [NeXus](https://www.nexusformat.org/) format with rich and standardized metadata annotation.

# Installation

## Prerequisites
- Python 3.9+
- pip

## Steps
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

# Documentation
Comprehensive documentation including several workflow examples can be found here:
https://opencompes.github.io/docs/sed/stable/


# Contributing
Users are welcome to contribute to the development of **sed-processor**. Information how to contribute, including how to install developer versions can be found in the [documentation](https://opencompes.github.io/docs/sed/stable/misc/contribution.html)

We would like to thank our contributors!

[![Contributors](https://contrib.rocks/image?repo=OpenCOMPES/sed)](https://github.com/OpenCOMPES/sed/graphs/contributors)


## License

sed-processor is licenced under the MIT license

Copyright (c) 2022-2024 OpenCOMPES

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

==============================
Contributing to sed
==============================

Welcome to the sed project, a collaboration of the Open Community of Multidimensional Photoemission Spectroscopy.

Whether you are a beamline scientist hoping to create a loader for your data, or would like to add a new feature to the project, we welcome your contributions.

This guide will walk you through the process of setting up your development environment, and the workflow for contributing to the project.


Getting Started
===============

1. **Clone the Repository:**

   - If you are a member of the repository, clone the repository to your local machine:

    .. code-block:: bash

        git clone https://github.com/OpenCOMPES/sed.git

   - If you are not a member of the repository, clone your fork of the repository to your local machine:

    .. code-block:: bash

        git clone https://github.com/yourusername/sed.git



2. **Install uv and Python:**
   - Ensure you have uv and Python 3.9, 3.10, 3.11 or 3.12 installed. You can install it e.g. using the following commands:

    .. code-block:: bash

        curl -LsSf https://astral.sh/uv/install.sh | sh
        uv python install 3.10

3. **Clone Repository:**

    .. code-block:: bash

        git clone https://github.com/OpenCOMPES/sed.git

4. **Set up virtual environment:**
   - Create a python virtual environment, and activate it. You can optionally select the python version, and set the path

    .. code-block:: bash

        uv venv -p=3.10 .venv
        source .venv/bin/activate


5. **Install Dependencies:**
   - Navigate to the project directory and install the project and its dependencies (including development ones) in "editable" mode using uv. Optionally, the jupyter notebook can be installed as well:

    .. code-block:: bash

        uv pip install -e .[dev,notebook]


Development Workflow
=====================

.. note::
   This guide assumes that you followed the previous steps and have uv, python and the package with dev dependencies installed on your machine.

1. **Install pre-commit hooks:** To ensure your code is formatted correctly, install pre-commit hooks:

    .. code-block:: bash

        pre-commit install


2. **Create a Branch:** Create a new branch for your feature or bug fix and make changes:

    .. code-block:: bash

        git checkout -b feature-branch


3. **Write Tests:** If your contribution introduces new features or fixes a bug, add tests to cover your changes.

4. **Run Tests:** To ensure no functionality is broken, run the tests:

    .. code-block:: bash

        pytest tests


5. **Commit Changes:** Commit your changes with a clear and concise commit message:

    .. code-block:: bash

        git commit -a -m "Your commit message"


6. **Push Changes:** Push your changes to your new branch:

    .. code-block:: bash

        git push origin feature-branch


7. **Open a Pull Request:** Open a pull request against the `main` branch of sed.

Pull Request Guidelines
=======================

Please give a brief description of the changes you have made in your pull request.
If your pull request fixes an issue, please reference the issue number in the pull request description.

Before your pull request can be merged, it must pass the following checks:

- **Linting Check**

- **Tests Check**

- **Code Review:** A maintainer will review your code and provide feedback if necessary.

- **Rebase with Main:** Ensure your branch is up-to-date with the latest changes from the `main` branch.

Once all checks are successful and your code is approved, it will be merged into the main branch.

Developing a Loader
===================
If you are developing a loader for your beamline, please follow the guidelines below.

1. **Create a Loader:**

   - Create a new loader in the `sed/loaders` directory.
   - The loader should be a subclass of `sed.loader.base.loader.BaseLoader` and implement a few methods. See :ref:`base_loader` for more information.
   - Give your class a `__name__` attribute, which is used to select the loader in user config files (See the generic loader for example).
   - At the end of your module, provide a `LOADER = YourNameLoader` variable, which is used to register that loader in the registry. See :ref:`loader_interface`.

2. **Write Tests:**

   - Write tests for your loader in the `tests/loaders` directory.
   - You can also include a small test data in the `tests/data` directory.

3. **Add Loader to Documentation:** Add your loader to the documentation in `docs/sed/loaders.rst`.

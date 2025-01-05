How to Maintain
===============

Documentation
-------------
**Build Locally:**

Users can generate documentation locally using the following steps:

1. **Install Dependencies:**

.. code-block:: bash

    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv python install 3.10

1. **Clone Repository:**

.. code-block:: bash

    git clone https://github.com/OpenCOMPES/sed.git

3. **Navigate to Repository:**

.. code-block:: bash

    cd sed

4. **Copy Tutorial Files:**

Doing this step will slow down the build process significantly. It also requires two datasets so 20 GB of free space is required.

.. code-block:: bash

    cp -r tutorial docs/
    cp -r sed/config docs/sed

5. **Create a virtual environment:**

.. code-block:: bash

    uv venv -p=3.10 .venv
    source .venv/bin/activate

6. **Install Dependencies:**

.. code-block:: bash

    uv pip install -e .[docs]

7. **Build Documentation:**

.. code-block:: bash

    sphinx-build -b html docs _build

8. **View Documentation:**

Open the generated HTML documentation in the `_build` directory.

**GitHub Workflow:**

The documentation workflow is designed to automatically build and deploy documentation. Additionally, maintainers of sed repository can manually trigger the documentation workflow from the Actions tab.
Here's how the workflow works:

1. **Workflow Configuration:**
   - The documentation workflow is triggered on push events to the main branch for specific paths and files related to documentation.
   - Manual execution is possible using the workflow_dispatch event from the Actions tab.

   .. code-block:: yaml

      on:
        push:
          branches: [ main ]
          paths:
            - sed/**/*
            - pyproject.toml
            - tutorial/**
            - .github/workflows/documentation.yml
        workflow_dispatch:

2. **Permissions:**
   - The workflow sets permissions for the GITHUB_TOKEN to allow deployment to GitHub Pages.
   - Permissions include read access to contents and write access to pages.

   .. code-block:: yaml

      permissions:
        contents: read
        pages: write
        id-token: write

3. **Concurrent Deployment:**
   - Only one concurrent deployment is allowed to prevent conflicts.
   - Future idea would be to have different deployment for different versions.
   - Runs queued between an in-progress run and the latest queued run are skipped.

   .. code-block:: yaml

      concurrency:
        group: "pages"
        cancel-in-progress: false

4. **Workflow Steps:**
   - The workflow is divided into two jobs: build and deploy.

     a. **Build Job:**
        - Sets up the build environment, checks out the repository, and installs necessary dependencies using uv.
        - Installs notebook dependencies and Pandoc.
        - Copies tutorial files to the docs directory.
        - Downloads RAW data for tutorials.
        - Builds Sphinx documentation.

     b. **Deploy Job:**
        - Deploys the built documentation to GitHub Pages repository.

5. **Manual Execution:**
   - To manually trigger the workflow, go to the Actions tab on GitHub.
   - Click on "Run workflow" for the "documentation" workflow.


Release
-------

**Creating a Release**

To create a release, follow these steps:

   a. **Create a Git Release on Github:**

      - On the "tags" page, select "releases", and press "Draft a new release".
      - At "choose a tag", type in the name of the new release tag. Make sure to have a **v** prefix in the tag name, e.g. **v0.1.10**.
      - Confirm creation of the tag, and press "Generate release notes". Edit the notes as appropriate (e.g. remove auto-generated update PRs).
      - Press "Publish release". This will create the new tag and release entry, and issue the build and upload to PyPI.

   b. **Check PyPI for the Published Package:**

      - Visit the PyPI page (https://pypi.org/project/sed-processor/).
      - Confirm that the new version (e.g., 0.1.10) has been published.

   c. **If you don't see update on PyPI:**

      - Visit the GitHub Actions page and monitor the Release workflow (https://github.com/OpenCOMPES/sed/actions/workflows/release.yml).
      - Check if errors occurred during the release process.


**Understanding the Release Workflow**

- *Release Job:*
    - This workflow is responsible for versioning and releasing the package.
    - A release job runs on every git release and publishes the package to PyPI.
    - The package version is dynamically obtained from the most recent git tag.

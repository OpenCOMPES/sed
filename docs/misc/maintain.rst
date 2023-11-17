How to maintain
===================================================

refer to github wiki as well

Building docs
===================================================

run
poetry shell
poetry install --with docs
sphinx-build -b html docs _build

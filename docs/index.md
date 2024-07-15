---
myst:
  html_meta:
    "description lang=en": |
      Top-level documentation for sed, with links to the rest
      of the site..
html_theme.sidebar_secondary.remove: true
---

# SED documentation

SED (Single Event Data Frame) is a collection of routines and utilities to handle photoelectron resolved datastreams.
It features lazy evaluation of dataframe processing using dask, numba-accelerated multi-dimensional binning, calibration and correction for trARPES (Time- and angle-resolved photoemission spectroscopy) datasets.
The package ensures provenance and FAIR data through metadata tracking, usage of the community defined NeXus format.

```{toctree}
:maxdepth: 2

user_guide/index

```

## Examples

Several example notebooks to demonstrate the functionality of SED for end-to-end data analysis workflows.

```{toctree}
:maxdepth: 2

workflows/index
```

## API

```{toctree}
:maxdepth: 2

sed/api
```


## Community and contribution guide

Information about the community behind this theme and how you can contribute.

```{toctree}
:maxdepth: 2

misc/contribution
```

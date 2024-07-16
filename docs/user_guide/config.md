# Configuration

The config module contains a mechanism to collect configuration parameters from various sources and configuration files, and to combine them in a hierarchical manner into a single, consistent configuration dictionary.
It will load an (optional) provided config file, or alternatively use a passed python dictionary as initial config dictionary, and subsequently look for the following additional config files to load:

* ``folder_config``: A config file of name :file:`sed_config.yaml` in the current working directory. This is mostly intended to pass calibration parameters of the workflow between different notebook instances.
* ``user_config``: A config file provided by the user, stored as :file:`.sed/config.yaml` in the current user's home directly. This is intended to give a user the option for individual configuration modifications of system settings.
* ``system_config``: A config file provided by the system administrator, stored as :file:`/etc/sed/config.yaml` on Linux-based systems, and :file:`%ALLUSERPROFILE%/sed/config.yaml` on Windows. This should provide all necessary default parameters for using the sed processor with a given setup. For an example for an mpes setup, see :ref:`example_config`
* ``default_config``: The default configuration shipped with the package. Typically, all parameters here should be overwritten by any of the other configuration files.

The config mechanism returns the combined dictionary, and reports the loaded configuration files. In order to disable or overwrite any of the configuration files, they can be also given as optional parameters (path to a file, or python dictionary).

## Default configuration settings

```{literalinclude} ../../sed/config/default.yaml
:language: yaml
```

## Example configuration file for mpes (METIS momentum microscope at FHI-Berlin)

```{literalinclude} ../../sed/config/mpes_example_config.yaml
:language: yaml
```

## Example configuration file for flash (HEXTOF momentum microscope at FLASH, Desy)

```{literalinclude} ../../sed/config/flash_example_config.yaml
:language: yaml
```

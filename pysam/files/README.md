# PySAM-DAO-Tk Package

https://github.com/NREL/lore/tree/develop/pysam

* Provides a wrapper around the DAO-Tk SAM library that groups together the C API functions by technology or financial model into modules.
* Includes error-checking, explicit input and output definition, and conversion between Python data types.
* Automatically assign default values to input parameters from SAM's default value database.
* Built-in documentation of models and parameters.


## Requirements
1. Python 3.5+, 64 bit
2. Operating system:
	- MacOSX 10.7+
	- Most Linux
	- Windows 7, x64
3. CMake 2.8


## Installing
1. PyPi:
	```
	pip install nrel-pysam-dao-tk
	```

May not be compatible with different builds of the CPython reference interpreter, and not with alternative interpreters such as PyPy, IronPython or Jython

## Usage Examples
- [Importing a SAM case](https://nrel-pysam.readthedocs.io/en/latest/Import.html)
- [Examples](https://github.com/NREL/pysam/blob/master/Examples)


## Citing this package

PySAM-DAO-Tk Version 1.0.1. National Renewable Energy Laboratory. Golden, CO. Accessed August 3, 2020. https://github.com/nrel/lore

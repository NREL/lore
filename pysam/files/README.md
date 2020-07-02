# DAO-Tk PySAM Package

https://github.com/NREL/dao-tk

* Provides a wrapper around the DAO-Tk SAM library that groups together the C API functions by technology or financial model into modules.
* Includes error-checking, explicit input and output definition, and conversion between Python data types.
* DAO-Tk PySAM modules are compatible with PySSC, which is included as a subpackage. PySSC is the original wrapper used by SAM's code generator.
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

DAO-Tk PySAM Version 1.0.0. National Renewable Energy Laboratory. Golden, CO. Accessed May 27, 2020. https://github.com/NREL/dao-tk

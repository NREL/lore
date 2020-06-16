# PySAM for the Design, Analysis, and Operations Toolkit (DAO-Tk)

PySAM is a Python library for calling modules in NREL's SAM software, and it's DAO-Tk version described here is for use with LORE.


## Installing the DAO-Tk Version of PySAM (in a fresh environment)
1. Create new environment, initializing with Python 3.7
```
conda create --name pysam_daotk python=3.7 -y
```

2. Activate that environment
```
conda activate pysam_daotk
```

3. Install dao-tk and dao-tk stubs
```
pip install nrel-pysam-dao-tk
pip install nrel-pysam-dao-tk-stubs
```

4. Uninstall nrel-pysam-stubs (shouldn't need to do this, but currently having to)
```
pip uninstall nrel-pysam-stubs
```


## Testing the DAO-Tk Version of PySAM in VS Code

1. Set configuration in VS Code
 
 open launch.json and add under 'console'
	```
	"env": {
		"pysam_daotk": "1",
	}
	```
	
2. Select interpreter in VS Code
	1. Ctrl-Shift-P  ->  Python: Select Interpreter
	2. Enter Interpreter Path, example:
	```
	C:\Users\mboyd\AppData\Local\Continuum\miniconda3\envs\pysam_daotk\python.exe
	```

3. Select environment in terminal
```
conda activate pysam_daotk
```

4. Test with mspt.py


## Creating a DAO-Tk Version of PySAM

### First time setup

1. Clone the [PySAM repository](https://github.com/NREL/pysam) into ...\sam_dev\pysam
2. Set the environment variable for pysam:
   <table>
   <tr><td>PYSAMDIR</td><td>...\sam_dev\pysam</td></tr>
   </table>
   
### First time and after
   
3. Checkout the DAO-Tk repo/branch(es)
4. Delete the contents of the .../sam_dev/build directory if it exists
5. Run CMake

     * Windows

         When running the CMake command to generate the visual studio solution ([step 7.4](https://github.com/NREL/SAM/wiki/Windows-Build-Instructions#7-run-cmake-to-generate-sam-vs-2019-project-files)), omit the setting that disables the API, as shown in the command below:
          ```
          cmake -G "Visual Studio 16 2019" -DCMAKE_CONFIGURATION_TYPES="Release" -DCMAKE_SYSTEM_VERSION=10.0 .. 
          ```
         Continue with the build in step 8. Doing an entire build including exporting the API takes upwards of 45 minutes, so please be patient.

     * Linux

         At the [install-lk-wex-ssc-then-sam](https://github.com/NREL/SAM/wiki/Linux-Build-Instructions#5-install-lk-wex-ssc-then-sam) step, change the `cmake` instruction to:

          ```
          cmake .. -DCMAKE_BUILD_TYPE=<Debug;Release> -Dskip_api=0 -DSAMAPI_EXPORT=1
          ```

     * Mac
    
         Modify the `cmake_setup_clean.sh` in the [Build SAM](https://github.com/NREL/SAM/wiki/Mac-Build-Instructions#build-sam) instructrions to enable `SAM_api` and to add `SAMAPI_EXPORT=1` by changing the last two lines:
          ```
          rm -rf $SSCDIR/build && mkdir $SSCDIR/build && cd $SSCDIR/build && cmake .. -DCMAKE_BUILD_TYPE=Release -Dskip_tests=1 -DSAMAPI_EXPORT=1 && cmake --build . -j8 --target ssc
          rm -rf $SAMNTDIR/build && mkdir $SAMNTDIR/build && cd $SAMNTDIR/build && cmake .. -DCMAKE_BUILD_TYPE=Release -Dskip_api=0 -DSAMAPI_EXPORT=1 && make -j8 
          ```
Once the build is finished, you should have the SAM and ssc libraries relevant to your system (`libssc.so` and `libSAM_api.so` on Unix and `ssc.dll, ssc.lib` and `SAM_api.dll, SAM_api.lib` on Windows) in the folder `pysam/files`

6. Open the solution file in .../sam_dev/build/ssc/
7. Unload the following projects:
	* TCSConsole
	* wex
	* SDKtool
	* lk
8. Batch build all
	* (There may be an error from build-time-make-directory)
9. Edit the version.py file
	* Increment the version (major.minor.patch)
	* Version must not equal any previous versions or PyPI will not let it on the repo
10. Edit RELEASE.md, adding the most recent changes
11. Copy version.py, README.md and RELEASE.md to .../pysam/files/, .../pysam/ and .../pysam/, respectively, overriding the existing files
12. Edit the arguments to setup() at the bottom of .../pysam/setup.py and ...pysam/stubs/setup.py. See corresponding files in this repo for examples.
	```
	name='NREL-PySAM-DAO-Tk'  [append '-stubs' for /stubs/setup.py]
	url='https://github.com/NREL/dao-tk'
	description="National Renewable Energy Laboratory's DAO-Tk Python Wrapper"		[append ', stub files' for /stubs/setup.py]
	author="Matthew-Boyd"
	author_email="matthew.boyd@nrel.gov"
	```
13. Create a /files/ directory in .../pysam/stubs/ and copy the version file to there
14. Activate a python virtual environment specific to this nrel-pysam-daotk version
15. Run build_win_daotk.bat to install the nrel-pysam-dao-tk package locally and create the wheel (.whl) files.
	* (There may be a couple test errors)
16. Upload package to PyPI
	1. Change directory to .../sam_dev/pysam/
	2. Create an account on [PyPi](https://pypi.org/) if you do not already have one
	2. Run:
	```
	pip install twine
	twine upload dist/*
	```
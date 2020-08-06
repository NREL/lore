@ECHO OFF

mkdir %SSCDIR%\..\build_pysam
cd %SSCDIR%\..\build_pysam

cmake -G "Visual Studio 16 2019" -DCMAKE_CONFIGURATION_TYPES="Release" -DCMAKE_SYSTEM_VERSION=10.0 -Dskip_tools=1 -Dskip_tests=1 ..
devenv /build Release system_advisor_model.sln

cd %PYSAMDIR%
echo y | rmdir build /s
echo y | del dist/*

FOR %%i IN (pysam_build_3.5, pysam_build_3.6 pysam_build_3.7, pysam_build_3.8) DO (
    call deactivate
    call activate %%i
    echo y | pip install -r tests\requirements.txt
    echo y | pip uninstall NREL-PySAM-DAO-Tk NREL-PySAM-DAO-Tk-stubs
    python setup.py install
    python .\stubs\setup.py install
    pytest -s tests
rem	if errorlevel 1 (
rem	   echo Error in Tests
rem	   exit /b %errorlevel%
rem	)
    python setup.py bdist_wheel
    python .\stubs\setup.py bdist_wheel
)

REM twine upload dist\*.whl
REM rmdir %SSCDIR%\..\build_pysam /s

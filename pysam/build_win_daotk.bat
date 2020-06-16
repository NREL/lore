@ECHO OFF

cd %PYSAMDIR%
echo y | rmdir build /s
echo y | del dist/*

FOR %%i IN (pysam_daotk) DO (
	call deactivate
    call activate %%i
    echo y | pip install -r tests/requirements.txt
    echo y | pip uninstall NREL-PySAM-DAO-Tk NREL-PySAM-DAO-Tk-stubs
    python setup.py install
    python ./stubs/setup.py install
    pytest -s tests
rem	if errorlevel 1 (
rem	   echo Error in Tests
rem	   exit /b %errorlevel%
rem	)
    python setup.py bdist_wheel
    python ./stubs/setup.py bdist_wheel
)



# Ensure pip and setuptools is available
python -m ensurepip

# Install virtualenv (no harm if already present)
python -m pip install virtualenv

# Create virtual env in the site folder
python -m virtualenv .

# Activate the new environment for the first time
Scripts\activate

# install the list of requirements
python -m pip install -r .\requirements.txt

# Get full path for the setup script
$SetupScriptPath = (Get-ChildItem -Filter '*setup.ps1').FullName

# Run Powershell as admin to be able to execute all of the setup script
Start-Process powershell.exe -Verb RunAs -ArgumentList '-noexit -Command',"&'$SetupScriptPath'"

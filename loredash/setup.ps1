# Loopback adapter not currently needed
#.\setup_loopback_adapter.ps1

## Import data to Django Models

Set-Location $PSScriptRoot

# Activate the environment
Scripts\activate

# Setup New Database
python manage.py migrate

## Only if there are no current migrations with data
Set-Location data
python import_data.py .

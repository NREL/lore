Start-Process powershell -ArgumentList "-executionpolicy bypass -File .\run_bokeh_server.ps1"
Start-Process powershell -ArgumentList "-executionpolicy bypass -File .\run_django.ps1"
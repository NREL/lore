$websocket = "10.10.10.10"
$address = "127.0.0.1"

Set-Location -Path ui\BokehApps

$bokeh_app_dirs = Get-ChildItem -Name '.\' -Directory
$bokeh_app_files = Get-ChildItem -Name '*.py' -File

bokeh serve $bokeh_app_dirs $bokeh_app_files --allow-websocket-origin $websocket --address $address
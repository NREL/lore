$websocket = '10.10.10.10'
$address = '127.0.0.1'

Set-Location -Path ui\BokehApps
$exclude_arr = @('theme', 'bokeh_utils')

$bokeh_app_dirs = Get-ChildItem -Name '.\' -Directory | Where-Object { $_ -notin $exclude_arr}
$bokeh_app_files = Get-ChildItem -Name '*.py' -File 

bokeh serve $bokeh_app_dirs $bokeh_app_files --allow-websocket-origin $websocket --address $address
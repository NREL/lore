# Note: if you edit this script, make sure to make the same changes to
#       `run_bokeh_server.sh`!

#$websocket = '10.10.10.10'
#$address = '127.0.0.1'

#Set-Location -Path ui\BokehApps
#$exclude_arr = @('theme', 'bokeh_utils')

#$bokeh_app_dirs = Get-ChildItem -Name '.\' -Directory | Where-Object { $_ -notin $exclude_arr}
#$bokeh_app_files = Get-ChildItem -Name '*.py' -File

#python -m bokeh serve $bokeh_app_dirs $bokeh_app_files --allow-websocket-origin $websocket --address $address

conda activate loredash

python -m bokeh serve `
    ./ui/BokehApps/dashboard_plot `
    ./ui/BokehApps/historical_dashboard_plot `
    ./ui/BokehApps/historical_solar_forecast `
    ./ui/BokehApps/market_plot `
    ./ui/BokehApps/solar_plot `
    ./ui/BokehApps/estimates_table.py `
    ./ui/BokehApps/probability_table.py `
    ./ui/BokehApps/sliders.py `
    ./ui/BokehApps/__init__.py `
    --allow-websocket-origin 127.0.0.1:8000 `
	--address 127.0.0.1 `
	--port 5006 `
	--check-unused-sessions 17000 `
    --unused-session-lifetime 600000

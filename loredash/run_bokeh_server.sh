#!/bin/bash
#
# Notes
# -----
#
# If you edit this script, make sure to make the same changes to
# `run_bokeh_server.ps1`!
#
# To kill this server on linux, run `$ ps -ef | grep bokeh` to identify the pid,
# then run `$ kill <pid>`.

conda activate loredash

python -m bokeh serve \
    ./ui/BokehApps/dashboard_plot \
    ./ui/BokehApps/historical_dashboard_plot \
    ./ui/BokehApps/historical_solar_forecast \
    ./ui/BokehApps/market_plot \
    ./ui/BokehApps/solar_plot \
    ./ui/BokehApps/estimates_table.py \
    ./ui/BokehApps/probability_table.py \
    ./ui/BokehApps/sliders.py \
    ./ui/BokehApps/__init__.py \
    --allow-websocket-origin 127.0.0.1:8000 --address 127.0.0.1 --port 5006

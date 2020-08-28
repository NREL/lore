@echo off

setlocal enabledelayedexpansion

set websocket=10.10.10.10
set address=127.0.0.1
set bokeh_dirs=

cd ui\BokehApps

for /d %%g in (.) do (
    if .!bokeh_dirs!==. (
        set bokeh_dirs=%%g 
    ) else (
        set bokeh_dirs=!bokeh_dirs! %%g
    )
)

bokeh serve !bokeh_files! --allow-websocket-origin %websocket% --address %address%

EXIT /B 0
# UI Dashboard

The UI dashboard is based on [Django](https://www.djangoproject.com/), which is a high-level Python web framework. In order to have interactive plots that are driven by Python code executed real-time, a charting server is needed. The open-source visualization library [Bokeh](https://bokeh.org/) is used for this purpose. Bokeh can be run a number of different ways, but running as a charting server is necessary to have Python-driven interative plots.

The Bokeh server can be run alongside Django same as before, including while debugging in Visual Studio Code. However, in order to do so on a local machine, with or without an additional web server (e.g., Apache), you must add another loopback address. Normally, on the production server, the Bokeh charting server will utilize the default loopback address of 127.0.0.1 and the web server will utilize the public facing IP address. On a local machine, however, the IP address cannot be repurposed in this way so another loopback address must be added, for either the the web server or Django's manage.py module. In order to get another loopback address, a loopback adapter must be installed.

For production testing, the web server [Waitress](https://docs.pylonsproject.org/projects/waitress/en/stable/) is used along with the [Nginx](https://www.nginx.com/) web server. The Nginx server is configured as a public-facing reverse proxy that passes the outside web requests to the Waitress server, but predominantly it is needed to serve the static files.

**Setup the Django Project (needed first time only)**
1. If using the Anaconda Python distribution platform (preferred default), setup PowerShell for use with Python.
  1. Open PowerShell and execute:
  	  ```
	  conda init
	  ```
  2. Close PowerShell
2. Setup and activate a dedicated virtual environment, install all needed Python packages, install a loopback adapter for the Bokeh server and populate the Django database.
  1. Navigate to the `\site` directory and run the following PowerShell script via:
     ```
	 .\init.ps1
	 ```
  2. Enter credentials when prompted in order to open up a new PowerShell instance with Administrator privileges
  3. Press Enter when asked to "Select which SQLite3 database to import to:"
  
  7. ***Very Important***: NREL computers will not allow a second network 'connection' at the same time as a WiFi connection (however, this may only be for NREL WiFi networks). The loopback adapter is seen as another network connection and thus WiFi will not work while this is present and enabled. To disable the loopback address and regain access to the NREL WiFi networks, start a command prompt with administrator privileges and run:
    ```
	netsh interface set interface "Ethernet 5" disable
	```
    This is for a loopback adapter named "Ethernet 5". To re-enable, just re-run the above command but change 'disable' to 'enable'. Verify the change by running `ipconfig /all`

**Run the Django Project**
1. If continuing from initial setup, in the same open PowerShell window, change to the `\site` directory:
   ```
   cd ..
   ```
2. Else if running the second time and after:
	1. Navigate to the `\site` directory and run the PowerShell script `setup.ps1`
	2. Press Enter when asked to "Select which SQLite3 database to import to:"
3. Run the script via `.\run_dashboard_app.ps1`
4. Open a browser and navigate to: `10.10.10.10`


# Dashboard Plots

The Dashboard stack currently uses Django and Bokeh both running on a server. These will both be run in production mode once on the local plant server.

The top of the Dashboard will be the same on the Dashboard, Forecast, and History Tabs. This header displays information that will update minutely. The five values indicate TES Charge State, Expected Solar Field Generation, Net Power Out, Annual Energy and Expected Revenue. The percentage values will represent the percent above or below their optimal amounts.

In the very top right of all pages there are indicators for the statuses of the Connection and the Model as well as the current time and date.

## Dashboard Home
![Dashboard](./media/README/main_dashboard_plot.png)
_Dashboard Plot_

The Dashboad shows the current information for the plant. It will show the Actual, Optimal, Scheduled, Field Operation Generated and Available values for the current daily window as well as the last 6, 12, 24, and 48 hour windows. The radio buttons on the top left will allow for the selection of a window, and the select buttons on the top right allow for multiple plots to be shown at the same time with the two axes denoted on the left and right, with the legend below denoting which axes apply to which plots.

## Forecasts
![Forecasts](./media/README/forecast_plots.png)

![Tables](./media/README/chart_examples.png)
_Forecasts Plots & Tables_

The Forecast tab will allow for the user to see the Market and Solar forecast data. Both plots have dropdowns for the window of time which will again come in 6, 12, 24, and 48 hour time blocks. The difference between these and the Dashboard plot time windows is that the windows are in the future from the current time.

The Probability and Estimates tables below provide information on weather predictions and startup and usage information for the current day, past week, and last 6 months.

## History
![History](./media/README/historical_plots.png)
_History Plots_

![History_Solar_Zoomed](./media/README/zoomed_in_example.png)
_Zoomed In Feature_

The history plots show the same data from the dashboard and solar plots, except these plots have sliders. The two sliders on bot plots allow for the user to change the date and the time window. The time window goes from -120 to 120 hours (&#177; 10 days).

## Navigation
To Navigate to the plots, only the first three tabs on the right hand side are active. These will take you to a main dahsboard page, forecast page, and a history page as noted in the sections above.

To zoom in, one can either pinch using touch screens, or use the scroll wheel to zoom. To pan, simply click and drag. The plots will only update their plot renders on the minute, or as the user make different selections. To reset the plot, simply double click.

Each plot has a pop-up that can be used to look closer at a given chart. In the bottom left hand of each chart pane, there is an expand icon. Clicking on the expand icon will enlarge the selected plot. When that plot is openned you can resume all of the same functionality as in the regulary view. When you are done, simply click the 'x' in the top right corner, or press the ESC key.

![Expand_Icon](./media/README/expand_icon.png)
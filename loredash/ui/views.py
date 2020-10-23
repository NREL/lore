from django.shortcuts import render
from django.http import HttpResponse, Http404, HttpResponseRedirect
from django.db import transaction

from bokeh.embed import server_session
from bokeh.util import token

# Other package imports
from datetime import datetime, timedelta
import matplotlib.pyplot as plt, mpld3
from io import StringIO
import pandas
from ui import apps

# pysam script
from ui import mspt

#global variables
PROGRESS_BAR_WIDTH = 160

# Create your views here.
#-------------------------------------------------------------
#-------------------------------------------------------------


def getPysamData():
    """Collect data from PySAM run"""
    pysam_output = mspt.get_pysam_data()
    pysam_output['time'] = list(pysam_output['time_hr'])
    pysam_output['time'] = [timedelta(hours=int(x)) for x in pysam_output['time']]
    pysam_output['time'] = list(map(lambda hr: hr + datetime(2010, 1, 1), pysam_output['time'])) # Jan 1, 2010 used because that is the start of our solar data
    return pysam_output

apps.pysam_output = getPysamData()

def getLiveStatusData():
    """Returns the last update time and connection and model statuses at the top of the main page."""

    return {
            'connection_status' : True,
            'model_status' : False,
            'last_refresh' : datetime.now(),
            }

def getLiveBarData():
    """Returns the data displayed in the 5 small boxes at the top of the main dashboard page."""

    pysam_output = apps.pysam_output

    ## Collects the data from the pysam_output stored on the server (as of now).
    ## This will update the bar on the hour (the frequency of the data entries in the weather file).

    # get current date and hour as well as yesterdays datetime 24 hours from the current
    current_time = datetime.today().replace(year=2010, minute=0, second=0, microsecond=0)
    prev_day = current_time - timedelta(days=1)

    live_data_index = {
        "time": 0,
        "e_ch_tes": 1, 
        "eta_therm": 2,
        "tou_value": 3, 
        "P_out_net": 4,
        "eta_field": 5}

    ## Live Data
    # Zip desired live data, and place into a list format
    live_data = list(zip(
        pysam_output["time"],
        pysam_output["e_ch_tes"],
        pysam_output["eta_therm"],
        pysam_output["tou_value"],
        pysam_output["P_out_net"],
        pysam_output["eta_field"]
    ))

    # Get data from both the previous day and current date and hour
    prev_days_data = list(filter(lambda t: t[0] == prev_day, live_data))[0]
    curr_live_data = list(filter(lambda t: t[0] == current_time, live_data))[0]

    # tes charge
    tes_charge = curr_live_data[live_data_index['e_ch_tes']]
    prev_tes_charge = prev_days_data[live_data_index['e_ch_tes']]
    # Get % change
    if prev_tes_charge != 0:
        tes_charge_pct_change = ((tes_charge - prev_tes_charge) / prev_tes_charge) * 100
    else:
        tes_charge_pct_change = ((tes_charge - prev_tes_charge) / 1) * 100

    # receiver thermal efficiency
    receiver_therm_eff = curr_live_data[live_data_index['eta_therm']] * 100
    prev_receiver_therm_eff = prev_days_data[live_data_index['eta_therm']] * 100
    # Get % change
    if prev_receiver_therm_eff != 0:
        receiver_therm_eff_pct_change = ((receiver_therm_eff - prev_receiver_therm_eff) / prev_receiver_therm_eff) * 100
    else:
        receiver_therm_eff_pct_change = ((receiver_therm_eff - prev_receiver_therm_eff) / 1) * 100

    # net power out
    net_power_out = curr_live_data[live_data_index['P_out_net']]
    prev_net_power_out = prev_days_data[live_data_index['P_out_net']]
    # Get % change
    if prev_net_power_out != 0:
        net_power_out_pct_change = ((net_power_out - prev_net_power_out) / prev_net_power_out) * 100
    else:
        net_power_out_pct_change = ((net_power_out - prev_net_power_out) / 1) * 100

    # CSP time of use value
    tou_value = curr_live_data[live_data_index['tou_value']]

    # field optical efficiency
    field_optical_eff = curr_live_data[live_data_index['eta_field']] * 100
    prev_field_optical_eff = prev_days_data[live_data_index['eta_field']] * 100
    # Get % change
    if prev_field_optical_eff != 0:
        field_optical_eff_pct_change = ((field_optical_eff - prev_field_optical_eff) / prev_field_optical_eff) * 100
    else:
        field_optical_eff_pct_change = ((field_optical_eff - prev_field_optical_eff) / 1) * 100

    # Export live data such that it can be used with the django template
    live_data = {
        "tes" : tes_charge,
        "tes_change" : tes_charge_pct_change,
        "receiver_therm_eff" : receiver_therm_eff,
        "receiver_therm_eff_change" : receiver_therm_eff_pct_change,
        "net_power_out" : net_power_out,
        "net_power_out_change" : net_power_out_pct_change,
        "tou_value" : tou_value,
        "field_optical_eff" : field_optical_eff,
        "field_optical_eff_change" : field_optical_eff_pct_change
    }

    return live_data

#-------------------------------------------------------------
def dashboard_view(request, *args, **kwargs):
    """
    main view for the dashboard
    """

    bokeh_server_url = "http://127.0.0.1:5006/dashboard_plot"
    server_script = server_session(None, session_id=token.generate_session_id(),
                                   url=bokeh_server_url)

    context = {"db_name" : "Dashboard",
               "db_script" : server_script,
               "live_data" : getLiveBarData(),
               **(getLiveStatusData())
              }

    return render(request, "ui/dashboard.html", context)

#-------------------------------------------------------------
def outlook_view(request, *args, **kwargs):


    bokeh_server_url = "http://127.0.0.1:5006/sliders"
    server_script = server_session(None, session_id=token.generate_session_id(),
                                   url=bokeh_server_url)
    context = {"graphname" : "Sliders",
               "server_script" : server_script
              }

    return render(request, "ui/outlook.html", context)

#-------------------------------------------------------------
def forecast_view(request, *args, **kwargs):

    from bokeh.embed import server_session

    market_url = "http://127.0.0.1:5006/market_plot"
    mkt_script = server_session(None, session_id=token.generate_session_id(),
                                    url=market_url)

    solar_url = "http://127.0.0.1:5006/solar_plot"
    solar_script = server_session(None, session_id=token.generate_session_id(),
                                    url=solar_url)

    probability_table_url = "http://127.0.0.1:5006/probability_table"
    probability_table_script = server_session(None, session_id=token.generate_session_id(),
                                    url=probability_table_url)
    
    estimates_table_url = "http://127.0.0.1:5006/estimates_table"
    estimates_table_script = server_session(None, session_id=token.generate_session_id(),
                                    url=estimates_table_url)
    
    context = {**(getLiveStatusData()),
               "mkt_script" : "Market Forecast",
               "mkt_script" : mkt_script,
               "solar_plot": "Solar Forecast",
               "solar_script": solar_script,
               "live_data": getLiveBarData(),
               "probability_table_script": probability_table_script,
               "estimates_table_script": estimates_table_script,

              }

    return render(request, "ui/forecast.html", context)

#-------------------------------------------------------------
def history_view(request, *args, **kwargs):
        
    hsf_url = "http://127.0.0.1:5006/historical_solar_forecast"
    hsf_server_script = server_session(None, session_id=token.generate_session_id(),
                                   url=hsf_url)
    hdbp_url = "http://127.0.0.1:5006/historical_dashboard_plot"
    hdbp_server_script = server_session(None, session_id=token.generate_session_id(),
                                   url=hdbp_url)

    context = {**(getLiveStatusData()),
               "hsf_plot_name": "Historical Solar Forecast Data",
               "hsf_script": hsf_server_script,
               "hdbp_plot_name": "Historical Dashboard Data",
               "hdbp_script": hdbp_server_script,
               "live_data": getLiveBarData()
              }
    return render(request, "ui/history.html", context)

#-------------------------------------------------------------
def maintenance_view(request, *args, **kwargs):
    context = {**(getLiveStatusData())}
    return render(request, "ui/maintenance.html", context)

#-------------------------------------------------------------
def settings_view(request, *args, **kwargs):
    context = {**(getLiveStatusData())}
    return render(request, "ui/settings.html", context)
#django imports
from django.shortcuts import render
from django.http import HttpResponse, Http404, HttpResponseRedirect
from django.db import transaction
# Model imports
from ui.models import pysam_output
# Other package imports
from datetime import datetime, timedelta
import matplotlib.pyplot as plt, mpld3
from io import StringIO
import pandas
#plot files imports
from fig import timeseries
#global variables
PROGRESS_BAR_WIDTH = 160

# Create your views here.
#-------------------------------------------------------------
#-------------------------------------------------------------

def getLiveStatusData():
    return {
            'connection_status' : True,
            'model_status' : False,
            'last_refresh' : datetime.now(),
            }

def getLiveBarData(request):

    # To be replaced with values from previous day
    prev_tes_charge_avg = 0
    prev_expected_solar_avg = 0
    prev_net_power_out_avg = 0
    prev_expected_revenue_avg = 0

    ## Live Data
    live_data = pysam_output

    # tes charge
    tes_charge_avg = sum(live_data["e_ch_tes"])/len(live_data["e_ch_tes"])
    tes_charge_change = tes_charge_avg > prev_tes_charge_avg

    # expected solar
    expected_solar_avg = sum(live_data["disp_qsfprod_expected"])/len(live_data["disp_qsfprod_expected"])
    expected_solar_change = expected_solar_avg > prev_expected_solar_avg

    # net power out
    net_power_out_avg = sum(live_data["P_out_net"])/len(live_data["P_out_net"])
    net_power_out_change = net_power_out_avg > prev_net_power_out_avg

    # annual energy
    annual_energy = live_data["annual_energy"]

    # expected Revenue
    expected_revenue_avg = sum(live_data["disp_rev_expected"])/len(live_data["disp_rev_expected"])
    expected_revenue_change = expected_revenue_avg > prev_expected_revenue_avg

    live_data = {

        "tes" : tes_charge_avg,
        "tes_change" : tes_charge_change,
        "expected_solar" : expected_solar_avg,
        "expected_solar_change" : expected_solar_change,
        "net_power_out" : net_power_out_avg,
        "net_power_out_change" : net_power_out_change,
        "annual_energy" : annual_energy,
        "expected_revenue" : expected_revenue_avg,
        "expected_revenue_change" : expected_revenue_change
    }

    return live_data

#-------------------------------------------------------------
def dashboard_view(request, *args, **kwargs):
    """
    main view for the dashboard
    """
    from bokeh.embed import server_session
    from bokeh.util import token

    bokeh_server_url = "http://127.0.0.1:5006/dashboard_plot"
    server_script = server_session(None, session_id=token.generate_session_id(),
                                   url=bokeh_server_url)

    context = {"db_name" : "Dashboard",
               "db_script" : server_script,
               "live_data" : getLiveBarData(request),
               **(getLiveStatusData())
              }

    return render(request, "dashboard.html", context)

#-------------------------------------------------------------
def outlook_view(request, *args, **kwargs):
    from bokeh.embed import server_session
    from bokeh.util import session_id


    bokeh_server_url = "http://127.0.0.1:5006/sliders"
    server_script = server_session(None, session_id=session_id.generate_session_id(),
                                   url=bokeh_server_url)
    context = {"graphname" : "Sliders",
               "server_script" : server_script
              }

    return render(request, "outlook.html", context)

#-------------------------------------------------------------
def forecast_view(request, *args, **kwargs):

    from bokeh.embed import server_session
    from bokeh.util import session_id

    market_url = "http://127.0.0.1:5006/market_plot"
    mkt_script = server_session(None, session_id=session_id.generate_session_id(),
                                    url=market_url)

    solar_url = "http://127.0.0.1:5006/solar_plot"
    solar_script = server_session(None, session_id=session_id.generate_session_id(),
                                    url=solar_url)

    probability_table_url = "http://127.0.0.1:5006/probability_table"
    probability_table_script = server_session(None, session_id=session_id.generate_session_id(),
                                    url=probability_table_url)
    
    estimates_table_url = "http://127.0.0.1:5006/estimates_table"
    estimates_table_script = server_session(None, session_id=session_id.generate_session_id(),
                                    url=estimates_table_url)
    
    context = {**(getLiveStatusData()),
               "mkt_script" : "Market Forecast",
               "mkt_script" : mkt_script,
               "solar_plot": "Solar Forecast",
               "solar_script": solar_script,
               "live_data": getLiveBarData(request),
               "probability_table_script": probability_table_script,
               "estimates_table_script": estimates_table_script,

              }

    return render(request, "forecast.html", context)

#-------------------------------------------------------------
def history_view(request, *args, **kwargs):

    from bokeh.embed import server_session
    from bokeh.util import session_id
        
    hsf_url = "http://127.0.0.1:5006/historical_solar_forecast"
    hsf_server_script = server_session(None, session_id=session_id.generate_session_id(),
                                   url=hsf_url)
    hdbp_url = "http://127.0.0.1:5006/historical_dashboard_plot"
    hdbp_server_script = server_session(None, session_id=session_id.generate_session_id(),
                                   url=hdbp_url)

    context = {**(getLiveStatusData()),
               "hsf_plot_name": "Historical Solar Forecast Data",
               "hsf_script": hsf_server_script,
               "hdbp_plot_name": "Historical Dashboard Data",
               "hdbp_script": hdbp_server_script,
               "live_data": getLiveBarData(request)
              }
    return render(request, "history.html", context)

#-------------------------------------------------------------
def maintenance_view(request, *args, **kwargs):
    context = {**(getLiveStatusData())}
    return render(request, "maintenance.html", context)

#-------------------------------------------------------------
def settings_view(request, *args, **kwargs):
    context = {**(getLiveStatusData())}
    return render(request, "settings.html", context)
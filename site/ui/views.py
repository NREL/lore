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

def getLiveBarData():

    current_time = datetime.today().replace(year=2010, minute=0, second=0, microsecond=0)
    prev_eod_time = current_time.replace(hour=23) - timedelta(days=1)

    live_data_index = {
        "time": 0,
        "e_ch_tes": 1, 
        "disp_qsfprod_expected": 2, 
        "P_out_net": 3,
        "disp_rev_expected": 4}

    ## Live Data
    live_data = list(zip(
        pysam_output["time"],
        pysam_output["e_ch_tes"],
        pysam_output["disp_qsfprod_expected"],
        pysam_output["P_out_net"],
        pysam_output["disp_rev_expected"]
    ))

    prev_eod_data = list(filter(lambda t: t[0] == prev_eod_time, live_data))[0]
    curr_live_data = list(filter(lambda t: t[0] == current_time, live_data))[0]

    # tes charge
    tes_charge = curr_live_data[live_data_index['e_ch_tes']]
    prev_tes_charge = prev_eod_data[live_data_index['e_ch_tes']]
    if prev_tes_charge != 0:
        tes_charge_pct_change = ((tes_charge - prev_tes_charge) / prev_tes_charge) * 100
    else:
        tes_charge_pct_change = ((tes_charge - prev_tes_charge) / 1) * 100

    # expected solar
    expected_solar = curr_live_data[live_data_index['disp_qsfprod_expected']]
    prev_expected_solar = prev_eod_data[live_data_index['disp_qsfprod_expected']]
    if prev_expected_solar != 0:
        expected_solar_pct_change = ((expected_solar - prev_expected_solar) / prev_expected_solar) * 100
    else:
        expected_solar_pct_change = ((expected_solar - prev_expected_solar) / 1) * 100

    # net power out
    net_power_out = curr_live_data[live_data_index['P_out_net']]
    prev_net_power_out = prev_eod_data[live_data_index['P_out_net']]
    if prev_net_power_out != 0:
        net_power_out_pct_change = ((net_power_out - prev_net_power_out) / prev_net_power_out) * 100
    else:
        net_power_out_pct_change = ((net_power_out - prev_net_power_out) / 1) * 100

    # annual energy
    annual_energy = pysam_output["annual_energy"]

    # expected Revenue
    expected_revenue = curr_live_data[live_data_index['disp_rev_expected']]
    prev_expected_revenue = prev_eod_data[live_data_index['disp_rev_expected']]
    if prev_expected_revenue != 0:
        expected_revenue_pct_change = ((expected_revenue - prev_expected_revenue) / prev_expected_revenue) * 100
    else:
        expected_revenue_pct_change = ((expected_revenue - prev_expected_revenue) / 1) * 100

    live_data = {

        "tes" : tes_charge,
        "tes_change" : tes_charge_pct_change,
        "expected_solar" : expected_solar,
        "expected_solar_change" : expected_solar_pct_change,
        "net_power_out" : net_power_out,
        "net_power_out_change" : net_power_out_pct_change,
        "annual_energy" : annual_energy,
        "expected_revenue" : expected_revenue,
        "expected_revenue_change" : expected_revenue_pct_change
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
               "live_data" : getLiveBarData(),
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
               "live_data": getLiveBarData(),
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
               "live_data": getLiveBarData()
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
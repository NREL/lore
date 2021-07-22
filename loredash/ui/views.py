from django.shortcuts import render
from bokeh.embed import server_session
from bokeh.util import token
import datetime

from mediation.models import TechData

#global variables
PROGRESS_BAR_WIDTH = 160

def getLiveStatusData():
    """Returns the last update time and connection and model statuses at the top of the main page."""

    return {
            'connection_status' : True,
            'model_status' : True,
            'last_refresh' : datetime.datetime.now(),
            }

def getLiveBarData():
    """Returns the data displayed in the 5 small boxes at the top of the dashboard pages.
    """
    # Latest data:
    try:
        queryset = TechData.objects.latest('timestamp')
        timestamp_latest = queryset.timestamp
        E_tes_charged = queryset.E_tes_charged                      # TES charge
        eta_tower_thermal = queryset.eta_tower_thermal              # Receiver/tower thermal efficiency
        W_grid_no_derate = queryset.W_grid_no_derate                # Net output power
        eta_field_optical = queryset.eta_field_optical              # Field optical efficiency
    except:
        E_tes_charged = None
        eta_tower_thermal = None
        W_grid_no_derate = None
        eta_field_optical = None

    # Data from 24 hr previous
    try:
        timestamp_prev = timestamp_latest - datetime.timedelta(days=1)
        queryset = TechData.objects.filter(timestamp=timestamp_prev)
        E_tes_charged_prev = queryset[0].E_tes_charged              # TES charge
        eta_tower_thermal_prev = queryset[0].eta_tower_thermal      # Receiver/tower thermal efficiency
        W_grid_no_derate_prev = queryset[0].W_grid_no_derate        # Net output power
        eta_field_optical_prev = queryset[0].eta_field_optical      # Field optical efficiency
    except:
        E_tes_charged = None
        eta_tower_thermal = None
        W_grid_no_derate = None
        eta_field_optical = None

    # Percent difference between latest and previous
    def perc_diff(latest, prev):
        if latest is None or prev is None:
            return None
        elif prev != 0:
            return (latest - prev) / prev * 100
        else:
            return latest - prev
    E_tes_charged_percdiff = perc_diff(E_tes_charged, E_tes_charged_prev)
    eta_tower_thermal_percdiff = perc_diff(eta_tower_thermal, eta_tower_thermal_prev)
    W_grid_no_derate_percdiff = perc_diff(W_grid_no_derate, W_grid_no_derate_prev)
    eta_field_optical_percdiff = perc_diff(eta_field_optical, eta_field_optical_prev)

    live_data = {
        "tes" : E_tes_charged * 1.e-3,                              # [MWht]
        "tes_change" : E_tes_charged_percdiff,
        "receiver_therm_eff" : eta_tower_thermal * 100.,            # [%]
        "receiver_therm_eff_change" : eta_tower_thermal_percdiff,
        "net_power_out" : W_grid_no_derate * 1.e-3,                 # [MWe]
        "net_power_out_change" : W_grid_no_derate_percdiff,
        "tou_value" : 5.0,
        "field_optical_eff" : eta_field_optical * 100.,             # [%]
        "field_optical_eff_change" : eta_field_optical_percdiff
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

    # probability_table_url = "http://127.0.0.1:5006/probability_table"
    # probability_table_script = server_session(None, session_id=token.generate_session_id(),
    #                                 url=probability_table_url)
    
    # estimates_table_url = "http://127.0.0.1:5006/estimates_table"
    # estimates_table_script = server_session(None, session_id=token.generate_session_id(),
    #                                 url=estimates_table_url)
    
    context = {**(getLiveStatusData()),
               "mkt_script" : "Market Forecast",
               "mkt_script" : mkt_script,
               "solar_plot": "Solar Forecast",
               "solar_script": solar_script,
               "live_data": getLiveBarData(),
            #    "probability_table_script": probability_table_script,
            #    "estimates_table_script": estimates_table_script,
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

#django imports
from django.shortcuts import render
from django.http import HttpResponse, Http404, HttpResponseRedirect
from django.db import transaction
# Model imports
from ui.models import DashboardSummaryItem, TimeSeriesEntry, TimeSeriesHighlight
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
def load_pysam_data(request):
    from ui import mspt

    request.session['pysam_output'] = mspt.get_pysam_data()
    print(request.session.keys())

def getLiveStatusData():
    return {
            'connection_status' : True,
            'model_status' : False,
            'last_refresh' : datetime.now(),
            }

#>>>>> temporary code to create necessary database objects
def _temp_populate_database():

    #-- dashboard summary items
    DashboardSummaryItem.objects.all().delete()
    data = []
                # name                      #units              #icon           #varname                #group      #max        #actual     #model
    data.append(["Daily production total",  "MWh<sub>e</sub>",  "flash.png",    "daily_production",     "summary",  110*14*.75, 110*14*.25, 110*14*.31   ])
    data.append(["Net power",               "MW<sub>e</sub>",   "tower.png",    "net_power",            "summary",  110,        95,         80          ])
    data.append(["Gross power",             "MW<sub>e</sub>",   "turbine.png",  "gross_power",          "summary",  120,        100,      111.9       ])
    data.append(["Thermal storage charge",  "MWh<sub>t</sub>",  "tank.png",     "tes_charge",           "summary",  110*12/.4,  110*6/.4,   110*7/.4    ])
    data.append(["Daily revenue",           "$",                "notes.png",    "daily_revenue",        "summary",  ] + [ d * 90 for d in data[0][-3:]])
    

    with transaction.atomic():
        for row in data:
            DSI = DashboardSummaryItem(
                name = row[0],
                units = row[1],
                icon = row[2],
                varname = row[3],
                group = row[4],
                baseline_max = row[5],
                actual = row[6],
                model = row[7],
            )
            DSI.save()
    #-- end dashboard summary items

    #-- hourly data
    TimeSeriesHighlight.objects.all().delete()

    # csvraw = [line.strip("\n").split(",")[0:2] for line in open("C:/Users/mwagner/Documents/NREL/software/dao-tk/web/site/fig/tsdata.csv",'r').readlines()[1:]]
    df = pandas.read_csv("./fig/tsdata.csv", index_col='timestamp', date_parser=timeseries._timestamp_parse, keep_date_col=True)
    hdr_pairs = [
        ["gross_actual", "gross_model"], 
        ["net_actual", "net_model"], 
        ["tes_actual", "tes_model"], 
        ["revenue_actual", "revenue_model"], 
        ["production_actual", "production_model"],
    ]

    with transaction.atomic():
        for pair in hdr_pairs:
            ky = pair[0].split("_")[0]
            #find associated highlight
            dsi = DashboardSummaryItem.objects.all().filter(varname__icontains=ky)[0]
            TSH = TimeSeriesHighlight(
                name = dsi.name,
                varname = dsi.varname,
                units = dsi.units,
                xaxis_label = "Time",
                yaxis_label = dsi.name + " " + dsi.units,
            )
            TSH.save()

            for timestamp,row in df[pair].iterrows():
                TSE = TimeSeriesEntry(
                        timestamp = timestamp,
                        data = "{:f},{:f}".format(row[pair[0]], row[pair[1]]),
                    )
                TSE.save()

                TSH.entries.add(TSE)
                TSH.save()

    #-- end hourly data

    return
#<<<<<<<<< end temporary code

#-------------------------------------------------------------
def dashboard_view(request, *args, **kwargs):
    """
    main view for the dashboard
    """
    from bokeh.embed import server_session
    from bokeh.util import token
    from django.contrib.sessions.backends.db import SessionStore

    if 'pysam_output' not in request.session:
        print("...loading PySAM data")
        load_pysam_data(request)

    bokeh_server_url = "http://127.0.0.1:5006/dashboard_plot"
    server_script = server_session(None, session_id=token.generate_session_id(),
                                   url=bokeh_server_url)

    context = {"db_name" : "Dashboard",
               "db_script" : server_script,
               "dashboard_data": request.session['pysam_output'],
               **(getLiveStatusData())
              }

    return render(request, "dashboard.html", context)

#-------------------------------------------------------------
def outlook_view(request, *args, **kwargs):
    from bokeh.embed import server_session
    from bokeh.util import session_id

    if 'pysam_output' not in request.session:
        print("...loading PySAM data")
        load_pysam_data(request)

    bokeh_server_url = "http://127.0.0.1:5006/sliders"
    server_script = server_session(None, session_id=session_id.generate_session_id(),
                                   url=bokeh_server_url)
    context = {"graphname" : "Sliders",
               "server_script" : server_script,
               "dashboard_data": request.session['pysam_output']
              }

    return render(request, "outlook.html", context)

#-------------------------------------------------------------
def forecast_view(request, *args, **kwargs):

    from bokeh.embed import server_session
    from bokeh.util import session_id

    if 'pysam_output' not in request.session:
        print("...loading PySAM data")
        load_pysam_data(request)

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
               "dashboard_data": request.session['pysam_output'],
               "probability_table_script": probability_table_script,
               "estimates_table_script": estimates_table_script,

              }

    return render(request, "forecast.html", context)

#-------------------------------------------------------------
def history_view(request, *args, **kwargs):

    from bokeh.embed import server_session
    from bokeh.util import session_id

    if 'pysam_output' not in request.session:
        print("...loading PySAM data")
        load_pysam_data(request)
        
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
               "dashboard_data": request.session['pysam_output']
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
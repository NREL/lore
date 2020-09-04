# Bokeh
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LinearAxis, DataRange1d, Legend, LegendItem, Band, HoverTool, PanTool, WheelZoomTool, CustomJS, NumeralTickFormatter, Span
from bokeh.models.widgets import Button, CheckboxGroup, RadioButtonGroup, Div, Select
from bokeh.palettes import Category20
from bokeh.layouts import column, row, WidgetBox, Spacer
from bokeh.themes import built_in_themes
from bokeh.themes import Theme
from bokeh.events import DoubleTap
from bokeh.io import curdoc
from tornado import gen
import sys
sys.path.append('theme')
import theme

# Data manipulation
import pandas as pd
import datetime
import numpy as np
from scipy.signal import savgol_filter
import re

# Asyncronous Access to Django DB
from ui.models import ForecastsMarketData as fmd
from threading import Thread
import queue

TIME_BOXES = {'NEXT_6_HOURS': 6,
              'NEXT_12_HOURS': 12,
              'NEXT_24_HOURS': 24,
              'NEXT_48_HOURS': 48
              }

data_labels = list(map(lambda col: col.name, fmd._meta.get_fields()))
current_datetime = datetime.datetime.now().replace(year=2010, second=0) # Eventually the year will be removed once live data is added

plus_minus_regx = re.compile('.*(?<!_minus)(?<!_plus)$')
base_data_labels = list(filter(plus_minus_regx.search, data_labels))
label_colors = {}
for i, data_label in enumerate(data_labels[2:]):
    label_colors.update({
        data_label: Category20[12][i]
    })
lines = {}

def getForecastMarketData(_range, queue):
    queryset = fmd.objects.filter(timestamp__range=_range).values_list(*(data_labels[1:]))
    df = pd.DataFrame.from_records(queryset)
    df.columns = data_labels[1:]
    queue.put(df)

def make_dataset(time_box):
    # Prepare data
    start_date = current_datetime
    end_date = current_datetime + datetime.timedelta(hours=TIME_BOXES[time_box])

    q = queue.Queue()

    # Get raw data
    thread = Thread(target=getForecastMarketData,
        args=((start_date, end_date),
        q))
    thread.start()
    thread.join()
    data_df = q.get()
    cds = ColumnDataSource(data_df)

    # Create Columns for lower and upper error bounds


    val_arr = np.array(cds.data['market_forecast'])
    val_minus_arr = np.array(cds.data['ci_minus'])/100
    val_plus_arr = np.array(cds.data['ci_plus'])/100
    cds.data['market_forecast_lower'] = list(\
        val_arr - np.multiply(val_arr, val_minus_arr))
    cds.data['market_forecast_upper'] = list(\
        val_arr + np.multiply(val_arr, val_plus_arr))

    return cds

# Styling for a plot
def style(p):
    # Title 
    p.title.align = 'center'
    p.title.text_font_size = '20pt'
    p.title.text_font = 'serif'

    # Axis titles
    p.xaxis.axis_label_text_font_size = '14pt'
    p.xaxis.axis_label_text_font_style = 'bold'
    p.yaxis.axis_label_text_font_size = '14pt'
    p.yaxis.axis_label_text_font_style = 'bold'

    # Tick labels
    p.xaxis.major_label_text_font_size = '12pt'
    p.yaxis.major_label_text_font_size = '12pt'

    return p

def make_plot(src): # Takes in a ColumnDataSource
    ## Create the plot

    # Add tools to plot
    wheel_zoom_tool = WheelZoomTool(maintain_focus=False)
    pan_tool = PanTool()
    hover_tool = HoverTool(
        tooltips=[
            ('Data', '$name'),
            ('Date', '$x{%a %b, %Y}'),
            ('Time', '$x{%R}'),
            ('Value', '$y')
        ],
        formatters={
            '$x':'datetime'
        }
    )

    plot = figure(
        tools=[wheel_zoom_tool, pan_tool, hover_tool], # this gives us our tools
        x_axis_type="datetime",
        toolbar_location = None,
        x_axis_label = None,
        y_axis_label = "Market Forecast",
        output_backend='webgl',
        sizing_mode='scale_width',
        aspect_ratio=2
        )

    # Set action to reset plot
    plot.js_on_event(DoubleTap, CustomJS(args=dict(p=plot), code="""
        p.reset.emit()
    """))

    plot.toolbar.active_drag = pan_tool
    plot.toolbar.active_scroll = wheel_zoom_tool

    plot.x_range.range_padding=0.005
    plot.x_range.range_padding_units="percent"

    # Set tick format to percentage
    plot.yaxis[0].formatter = NumeralTickFormatter(format='0.00%')

    plot.line( 
        x='timestamp',
        y='market_forecast',
        line_color = 'green', 
        line_alpha = 1.0, 
        line_width=3,
        source=src,
        name='Market Forectast'
        )

    band = Band(
        base='timestamp',
        lower='market_forecast_lower',
        upper='market_forecast_upper',
        source=src,
        level = 'underlay',
        fill_color = 'green',
        fill_alpha=0.4,
        line_width=1, 
        line_alpha=0.0,
        )

    plot.add_layout(band)
    # styling
    plot = style(plot)

    return plot

def update_points(attr, old, new):
    # Update plots when widgets change

    # Get updated time block information
    active_time_window = time_window.options.index(time_window.value)
    time_box = list(TIME_BOXES.keys())[active_time_window]

    # Update data
    new_src = make_dataset(time_box)
    src.data.update(new_src.data)

@gen.coroutine
def live_update():
    ## Do a live update on the minute
    global current_datetime

    new_current_datetime = datetime.datetime.now().replace(year=2010, second=0) # Until live data is being used
    
    q = queue.Queue()

    # Get updated time block information
    active_time_window = time_window.options.index(time_window.value)
    time_box = list(TIME_BOXES.keys())[active_time_window]
    time_delta = datetime.timedelta(hours=TIME_BOXES[time_box])

    # Current Data
    thread = Thread(target=getForecastMarketData, 
        args=((current_datetime + time_delta, new_current_datetime  + time_delta), 
            q))
    thread.start()
    thread.join()
    current_data_df = q.get()

    # Add _lower and _upper columns for plotting
  
    val_arr = np.array(current_data_df['market_forecast'])
    val_minus_arr = np.array(current_data_df['ci_minus'])/100
    val_plus_arr = np.array(current_data_df['ci_plus'])/100
    current_data_df['market_forecast_lower'] = list(\
        val_arr - np.multiply(val_arr, val_minus_arr))
    current_data_df['market_forecast_upper'] = list(\
        val_arr + np.multiply(val_arr, val_plus_arr))
    current_data_df.index.name = 'index'

    src.stream(current_data_df)
    df_temp = src.to_df().drop([0]).drop('index', axis=1)
    src.data.update(ColumnDataSource(df_temp).data)

    current_datetime = new_current_datetime
    


# Create widget layout

time_window_init = "Next 24 Hours"
time_window = Select(
    options=["Next 6 Hours", "Next 12 Hours", "Next 24 Hours", "Next 48 Hours"], 
    value=time_window_init,
    width=150)
time_window.on_change('value', update_points)

src = make_dataset('NEXT_24_HOURS')

plot = make_plot(src)

widgets = row(
    Spacer(width_policy='max'),
    time_window
    )

title = Div(text="""<h3>Market Forecast</h3>""")

layout = column(
    row(
        title,
        widgets,
        width_policy='max'),
    plot,
    sizing_mode='stretch_both'
)

curdoc().add_root(layout)
curdoc().add_periodic_callback(live_update, 60000)
curdoc().title = "Market Forecast Plot"
curdoc().theme=Theme(json=theme.json)
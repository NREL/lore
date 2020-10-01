# Bokeh
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LinearAxis, DataRange1d, Legend, LegendItem, Band, HoverTool, WheelZoomTool, PanTool, CustomJS, Span
from bokeh.models.widgets import RadioButtonGroup, CheckboxButtonGroup, Div, Select
import colorcet as cc
from bokeh.layouts import column, row, WidgetBox, Spacer
from bokeh.themes import Theme, built_in_themes
from bokeh.events import DoubleTap
from bokeh.io import curdoc
from tornado import gen
import sys
sys.path.append('theme')
sys.path.append('bokeh_utils')
import theme
import bokeh_utils as butils


# Data manipulation
import pandas as pd
import datetime
import numpy as np
import re
from scipy.signal import savgol_filter

# Asyncronous access to Django DB
from ui.models import ForecastsSolarData as fsd
from threading import Thread
import queue

TIME_BOXES = {'NEXT_6_HOURS': 6,
              'NEXT_12_HOURS': 12,
              'NEXT_24_HOURS': 24,
              'NEXT_48_HOURS': 48
              }

data_labels = list(map(lambda col: col.name, fsd._meta.get_fields()))
current_datetime = datetime.datetime.now().replace(year=2010, second=0) # Eventually the year will be removed once live data is added

plus_minus_regx = re.compile('.*(?<!_minus)(?<!_plus)$')
base_data_labels = list(filter(plus_minus_regx.search, data_labels))
label_colors = {col+'_color': cc.fire[(i+1)*40] for i,col in enumerate(base_data_labels[2:])}
lines = {}
bands = {}

def getForecastSolarData(_range, queue):
    queryset = fsd.objects.filter(timestamp__range=_range).values_list(*(data_labels[1:]))
    df = pd.DataFrame.from_records(queryset)
    df.columns = data_labels[1:]
    queue.put(df)

def make_dataset(time_box, distribution):
    # Prepare data
    start_date = current_datetime
    end_date = current_datetime + datetime.timedelta(hours=TIME_BOXES[time_box])

    q = queue.Queue()

    # Get raw data
    thread = Thread(target=getForecastSolarData,
        args=((start_date, end_date),
        q))
    thread.start()
    thread.join()
    data_df = q.get()
    cds = ColumnDataSource(data_df)

    # Create Columns for lower and upper error bounds

    for col_name in base_data_labels[3:]:
        val_arr = np.array(cds.data[col_name])
        val_minus_arr = np.array(cds.data[col_name+'_minus'])/100
        val_plus_arr = np.array(cds.data[col_name+'_plus'])/100
        cds.data[col_name+'_lower'] = list(\
            val_arr - np.multiply(val_arr, val_minus_arr))
        cds.data[col_name+'_upper'] = list(\
            val_arr + np.multiply(val_arr, val_plus_arr))

    if distribution == "Smoothed":
        window, order = 51, 3
        for label in cds.column_names[2:]:
            cds.data[label] = savgol_filter(cds.data[label], window, order)
    
    return cds

def make_plot(src): # Takes in a ColumnDataSource
    ## Create the plot

    # Setup plot tools
    wheel_zoom_tool = WheelZoomTool(maintain_focus=False)
    pan_tool = PanTool()
    hover_tool = HoverTool(
        line_policy='nearest',
        tooltips=[
            ('Data','$name'),
            ('Date', '$x{%a %b, %Y}'),
            ('Time', '$x{%R}'),
            ('Value', '$y W/m^2')
        ],
        formatters={
            '$x':'datetime'
        }
    )

    plot = figure(
        tools=[wheel_zoom_tool, pan_tool, hover_tool], # this gives us our tools
        x_axis_type="datetime",
        width=650,
        height=525,
        sizing_mode='stretch_both',
        toolbar_location = None,
        x_axis_label = None,
        y_axis_label = "Power (W/m^2)",
        output_backend='webgl'
        )

    # Set action to reset plot
    plot.js_on_event(DoubleTap, CustomJS(args=dict(p=plot), code="""
        p.reset.emit()
    """))

    plot.toolbar.active_drag = pan_tool
    plot.toolbar.active_scroll = wheel_zoom_tool

    plot.x_range.range_padding=0.005
    plot.x_range.range_padding_units="percent"

    legend = Legend(orientation='vertical', location='top_left', spacing=10)
    
    for label in base_data_labels[2:]:

        legend_label = butils.col_to_title_upper(label)
        lower_upper_regex = re.compile(label+'(_plus|_minus)')

        if len(list(filter(lower_upper_regex.search, data_labels))):
            bands[label] = Band(
                base='timestamp',
                lower= label + '_lower',
                upper= label + '_upper',
                source=src,
                level = 'underlay',
                fill_alpha=0.2,
                fill_color=label_colors[label+'_color'],
                line_width=1, 
                line_alpha=0.0,
                visible = label in [butils.title_to_col(plot_select.labels[i]) for i in plot_select.active],
                name = legend_label,
                )
            plot.add_layout(bands[label])
            lines[label] = plot.line( 
                x='timestamp',
                y=label,
                line_color = label_colors[label+'_color'], 
                line_alpha = 1.0,
                line_width=3,
                source=src,
                visible = label in [butils.title_to_col(plot_select.labels[i]) for i in plot_select.active],
                name = legend_label, 
                )
            legend_item = LegendItem(label=legend_label, renderers=[lines[label]])
            legend.items.append(legend_item)
        else:
            lines[label] = plot.line( 
                x='timestamp',
                y=label,
                line_color = label_colors[label+'_color'], 
                line_alpha = 1.0,
                line_width=3,
                source=src,
                visible = label in [butils.title_to_col(plot_select.labels[i]) for i in plot_select.active],
                name = legend_label,
                )
            legend_item = LegendItem(label=legend_label, renderers=[lines[label]])
            legend.items.append(legend_item)

    # styling
    plot = butils.style(plot)

    plot.add_layout(legend, 'left')

    return plot

def update_lines(attr, old, new):
    # Update visible lines
    selected_labels = [plot_select.labels[i] for i in plot_select.active]

    for label in lines.keys():
        label_name = butils.col_to_title_upper(label)
        lines[label].visible = label_name in selected_labels
        if label in bands.keys():
            bands[label].visible = lines[label].visible

def update_points(attr, old, new):
    # Update plots when widgets change

    # Get updated time block information
    active_time_window = time_window.options.index(time_window.value)
    time_box = list(TIME_BOXES.keys())[active_time_window]

    # Update data
    new_src = make_dataset(time_box, distribution_select.value)
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
    thread = Thread(target=getForecastSolarData, 
        args=((current_datetime + time_delta, new_current_datetime + time_delta), 
            q))
    thread.start()
    thread.join()
    current_data_df = q.get()

    # Add _lower and _upper columns for plotting
    for col_name in base_data_labels[3:]:
        val_arr = np.array(current_data_df[col_name])
        val_minus_arr = np.array(current_data_df[col_name+'_minus'])/100
        val_plus_arr = np.array(current_data_df[col_name+'_plus'])/100
        current_data_df[col_name+'_lower'] = list(\
            val_arr - np.multiply(val_arr, val_minus_arr))
        current_data_df[col_name+'_upper'] = list(\
            val_arr + np.multiply(val_arr, val_plus_arr))

    src.stream(current_data_df)
    df_temp = src.to_df().drop([0]).drop('index', axis=1)
    src.data.update(ColumnDataSource(df_temp).data)

    current_datetime = new_current_datetime

# Create widgets
# Create Radio Button Group Widget
time_window_init = "Next 24 Hours"
time_window = Select(
    options=["Next 6 Hours", "Next 12 Hours", "Next 24 Hours", "Next 48 Hours"], 
    value=time_window_init,
    width=135)
time_window.on_change('value', update_points)

# Create Checkbox Select Group Widget
labels_list = [butils.col_to_title_upper(label) for label in base_data_labels[2:]]
plot_select = CheckboxButtonGroup(
    labels = labels_list,
    active = [0],
    width_policy='min'
)
plot_select.on_change('active', update_lines)

distribution = 'Discrete'
distribution_select = Select(
    value=distribution,
    options=['Discrete', 'Smoothed'],
    width=120
    )
distribution_select.on_change('value', update_points)


title = Div(text="""<h3>Solar Forecast</h3>""")

# Set initial plot information
initial_plots = [butils.title_to_col(plot_select.labels[i]) for i in plot_select.active]

src = make_dataset('NEXT_24_HOURS', distribution)

plot = make_plot(src)

# Setup Widget Layouts

widgets = column(
    row(
        title,
        row(
            Spacer(width_policy='max'),
            plot_select
        ),
        width_policy='max'

    ),
    row(
        Spacer(width_policy='max'),
        time_window,
        distribution_select),
        width_policy='max'
    
)

layout = column(
    row(widgets),
    Spacer(height=10),
    plot,
    sizing_mode='stretch_width',
    width_policy='max'
)

# Show to current document/page
curdoc().add_root(layout)
curdoc().add_periodic_callback(live_update, 60000)
curdoc().title = "Solar Forecast Plot"
curdoc().theme = Theme(json=theme.json)
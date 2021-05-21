import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# Bokeh
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LinearAxis, DataRange1d, Legend, LegendItem, PanTool, WheelZoomTool, HoverTool, CustomJS, Span
from bokeh.models.widgets import CheckboxButtonGroup, RadioButtonGroup, Div, DateSlider, Slider, Button, DatePicker
from bokeh.layouts import column, row, WidgetBox, Spacer
from bokeh.themes import Theme
from bokeh.events import DoubleTap
from bokeh.io import curdoc
import bokeh_utils.bokeh_utils as butils

from tornado import gen
import colorcet as cc
import theme.theme as theme

# Data manipulation
import pandas as pd
import sqlite3
import datetime
import re

# Asyncronous Access to Django DB
from mediation.models import TechData as dd
from threading import Thread
import queue

# From the TechData model table
PLOT_LABELS_FOR_DATA_COLS = {
    'Timestamp': 'timestamp',
    'Actual [MWe]': 'W_grid_no_derate',
    'Optimal [MWe]': 'W_grid_with_derate',
    'Scheduled [MWe]': 'W_grid_with_derate',            # W_grid_with_derate is not really the scheduled value
    'Field Op. Generated [MWt]': 'Q_tower_incident',
    'Field Op. Available [MWt]': 'Q_field_incident'
}

# Corresponds to PLOT_LABELS_FOR_DATA_COLS
CURRENT_DATA_COLS = [0, 1, 4, 5]
FUTURE_DATA_COLS = [0, 2, 3]


data_labels = list(PLOT_LABELS_FOR_DATA_COLS.keys())
data_labels_no_units = [re.sub(' \[.*\]$', '', label) for label in data_labels]
data_columns = list(PLOT_LABELS_FOR_DATA_COLS.values())
label_colors = {col+'_color': cc.glasbey_cool[i] for i,col in enumerate(data_labels[1:])}

current_datetime = datetime.datetime.now().replace(second=0, microsecond=0)
lines = {}

def getDashboardData(_range, _values_list, queue):
    queryset = dd.objects.filter(timestamp__range=_range).values_list(*_values_list)
    df = pd.DataFrame.from_records(queryset)
    if not df.empty:
        df.columns=_values_list
    else:
        df = pd.DataFrame(columns=_values_list)
    queue.put(df)

def getTimeRange(queue):
    times = dd.objects.values_list('timestamp')
    start_date = times.order_by('timestamp').first()[0]
    end_date = current_datetime
    queue.put((start_date, end_date))

def make_dataset(_start_date, _end_date):
    # Prepare data
    if _end_date > current_datetime and _start_date >= current_datetime or _end_date == _start_date:
        return src

    q = queue.Queue()

    # Get Current Data
    thread = Thread(target=getDashboardData,
        args=((_start_date, _end_date),
            data_columns,
            q))
    thread.start()
    thread.join()
    current_data_df = q.get()
    cds = ColumnDataSource(current_data_df)

    return cds

def make_plot(src): # (Source Data)
    ## Create the plot

    # Setup plot tools
    wheel_zoom_tool = WheelZoomTool(maintain_focus=False)
    pan_tool = PanTool()
    hover_tool = HoverTool(
        tooltips=[
            ('Data','$name'),
            ('Date', '$x{%a %b, %Y}'),
            ('Time', '$x{%R}'),
            ('Value', '$y')
        ],
        formatters={
            '$x':'datetime'
        }
    )

    plot = figure(
        tools=[hover_tool, wheel_zoom_tool, pan_tool], # this gives us our tools
        x_axis_type="datetime",
        width=650,
        height=525,
        sizing_mode='stretch_both',
        toolbar_location = None,
        x_axis_label = None,
        y_axis_label = "Power (MWe)",
        output_backend='webgl',
        )
    # Set action to reset plot
    plot.js_on_event(DoubleTap, CustomJS(args=dict(p=plot), code="""
        p.reset.emit()
    """))

    plot.toolbar.active_drag = pan_tool
    plot.toolbar.active_scroll = wheel_zoom_tool

    plot.x_range.range_padding=0.005
    plot.x_range.range_padding_units="percent"

    plot.extra_y_ranges = {"mwt": DataRange1d()}
    plot.add_layout(LinearAxis(y_range_name="mwt", axis_label="Power (MWt)"), 'right')
    
    legend = Legend(orientation='horizontal', location='top_center', spacing=10)
    
    # Add current time vertical line
    current_line = Span(
        location=current_datetime,
        dimension='height',
        line_color='white',
        line_dash='dashed',
        line_width=2
    )
    plot.add_layout(current_line)

    for data_label, data_column in PLOT_LABELS_FOR_DATA_COLS.items():
        if 'Timestamp' in data_label: continue
        data_label_no_unit = re.sub(' \[.*\]$', '', data_label)
        if 'Field' in data_label:
            y_range_name = 'mwt'
            level = 'underlay'
        else:
            y_range_name = 'default'
            level= 'glyph' if 'Actual' in data_label else 'underlay'

        lines[data_label] = plot.line( 
            x = PLOT_LABELS_FOR_DATA_COLS['Timestamp'],
            y = data_column,
            line_color = label_colors[data_label + '_color'], 
            line_alpha = 1.0, 
            hover_line_color = label_colors[data_label + '_color'],
            y_range_name=y_range_name,
            level=level,
            source = src,
            line_width=3,
            visible = data_label_no_unit in [plot_select.labels[i] for i in plot_select.active],
            name = data_label
            )

        if 'Field' in data_label:
            plot.extra_y_ranges['mwt'].renderers.append(lines[data_label])
        else:
            plot.y_range.renderers.append(lines[data_label])

        legend_item = LegendItem(label=data_label.replace('Operation', 'Op.'), renderers=[lines[data_label]])
        legend.items.append(legend_item)
    
    # styling
    plot = butils.style(plot)
    legend.label_text_font_size = '11px'

    plot.add_layout(legend, 'below')

    return plot

def update_lines(attr, old, new):
    # Update visible lines
    selected_labels = [plot_select.labels[i] for i in plot_select.active]
    
    for label in lines.keys():
        label_name_no_units = re.sub(' \[.*\]$', '', butils.col_to_title(label))
        lines[label].visible = label_name_no_units in selected_labels

def update_points(attr, old, new):
    # Update range when sliders move and update button is clicked
    range_start, range_end = butils.get_update_range(date_slider, date_span_slider, current_datetime)
    new_src = make_dataset(range_start, range_end)
    src.data.update(new_src.data)

@gen.coroutine
def live_update():
    ## Do a live update on the minute
    global current_datetime

    new_current_datetime = datetime.datetime.now().replace(second=0, microsecond=0)

    # Change location of timeline
    getattr(plot, 'center')[2].location = new_current_datetime

    _, range_end = butils.get_update_range(date_slider, date_span_slider, current_datetime)
    if current_datetime <= range_end:
        q = queue.Queue()

        # Current Data
        thread = Thread(target=getDashboardData, 
            args=((current_datetime, new_current_datetime), 
                data_columns,
                q))
        thread.start()
        thread.join()
        current_data_df = q.get()
        
        src.stream(current_data_df)
        df_temp = src.to_df().drop([0]).drop('index', axis=1)
        src.data.update(ColumnDataSource(df_temp).data)

    current_datetime = new_current_datetime
    date_slider.end = current_datetime.date() + datetime.timedelta(days=1)


# Create widget layout
# Create Checkbox Select Group Widget
labels_list = [label for label in data_labels_no_units if label != 'Timestamp']
labels_list = list(map(lambda label: label.replace('Operation', 'Op.'), labels_list))
plot_select = CheckboxButtonGroup(
    labels = labels_list,
    active = [0],
    width_policy='min'
)
plot_select.on_change('active', update_lines)

# Create Date Slider
# Get start and end date in table
q = queue.Queue()
thread = Thread(target=getTimeRange, args=(q,))
thread.start()
thread.join()
(start_date, end_date) = q.get()

date_slider = DateSlider(
    title='Date',
    start=start_date, 
    end=end_date.date() + datetime.timedelta(days=1), 
    value=end_date,
    step=1, 
    width=150)
date_slider.on_change('value_throttled', update_points)

# Create Date Range Slider
date_span_slider = Slider(
    title='Time Span (Hours)', 
    start=-240, 
    end=240, 
    value=-24, 
    step=4, 
    width=150)
date_span_slider.on_change('value_throttled', update_points)

title = Div(text="""<h3>Dashboard Data</h3>""")

# Set initial plot information
initial_plots = [butils.title_to_col(plot_select.labels[i]) for i in plot_select.active]

delta_init = datetime.timedelta(hours=24)
src = make_dataset(current_datetime - delta_init, current_datetime)
plot = make_plot(src)

# Setup Widget Layouts

layout = column(
    row(
        title,
        Spacer(width_policy='max'),
        date_slider,
        date_span_slider
    ),
    row(
        Spacer(width_policy='max'),
        plot_select
    ),
    Spacer(height=10),
    plot,
    sizing_mode='stretch_width',
)

curdoc().add_root(layout)
curdoc().add_periodic_callback(live_update, 60000)
curdoc().title = "Historical Dashboard Plot"
curdoc().theme = Theme(json=theme.json)
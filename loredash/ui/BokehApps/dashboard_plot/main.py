import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# Bokeh
from bokeh.plotting import figure
from bokeh.models import Span, ColumnDataSource, LinearAxis, DataRange1d, Legend, LegendItem, PanTool, WheelZoomTool, HoverTool, CustomJS
from bokeh.models.widgets import Button, CheckboxButtonGroup, RadioButtonGroup
from bokeh.layouts import column, row, WidgetBox, Spacer
from bokeh.themes import Theme
from bokeh.io import curdoc
from bokeh.events import DoubleTap
import bokeh_utils.bokeh_utils as butils

import colorcet as cc
from tornado import gen
import theme.theme as theme

# Data manipulation
import pandas as pd
import datetime
import re

# Asyncronous access to Django DB
from mediation.models import TechData as dd
import threading
import queue

TIME_BOXES = {
    'Today': 1,
    'Last 6 Hours': 6,
    'Last 12 Hours': 12,
    'Last 24 Hours': 24,
    'Last 48 Hours': 48
}

# From the TechData model table
PLOT_LABELS_FOR_DATA_COLS = {
    'Timestamp': 'timestamp',
    'Actual [MWe]': 'W_grid_no_derate',
    'Optimal [MWe]': 'W_grid_with_derate',
    'Scheduled [MWe]': 'W_grid_with_derate',            # W_grid_with_derate is not really the scheduled value
    'Field Op. Generated [MWt]': 'Q_tower_incident',
    'Field Op. Available [MWt]': 'Q_field_incident'
}

CURRENT_DATA_COLS = [
    'timestamp',
    'W_grid_no_derate',
    'Q_tower_incident',
    'Q_field_incident',
]

FUTURE_DATA_COLS = [
    'timestamp',
    'W_grid_with_derate',
    'W_grid_with_derate',
]

def _strip_unit(label):
    return re.sub(' \[.*\]$', '', label)

current_datetime = datetime.datetime.now().replace(second=0, microsecond=0)

# A global variable to hold all of the plot lines!
PLOT_LINES = {}

def getDashboardData(queue, date_range, columns):
    """
    Get the dashboard data corresponding `columns` over the `date_range` given
    as the tuple `(date_start, date_stop)`. Store the result in `queue`.
    """
    rows = dd.objects.filter(timestamp__range=date_range).values_list(*columns)
    df = pd.DataFrame.from_records(rows)
    if not df.empty:
        df.columns = columns
    else:
        df = pd.DataFrame(columns=columns)
    queue.put(df)
    return

def _getDashboardData(queue, start_date, end_date, columns):
    "A helper function that wraps the threading calls."
    thread = threading.Thread(
        target=getDashboardData,
        args=(queue, (start_date, end_date), columns),
    )
    thread.start()
    thread.join()
    return queue.get()

def make_dataset(time_box):
    # Prepare data
    start_date = current_datetime.date()
    end_date = current_datetime
    pred_end_date = current_datetime.date() + datetime.timedelta(days=1)

    if time_box != 'Today':
        start_date = current_datetime - datetime.timedelta(hours=TIME_BOXES[time_box])
        pred_end_date = current_datetime

    q = queue.Queue()

    # Current Data
    current_data_df = _getDashboardData(
        q,
        start_date,
        end_date,
        CURRENT_DATA_COLS,
    )
    current_cds = ColumnDataSource(current_data_df)

    # Future Data
    predictive_data_df = _getDashboardData(
        q,
        start_date,
        pred_end_date,
        FUTURE_DATA_COLS,
    )
    predictive_cds = ColumnDataSource(predictive_data_df)
    return predictive_cds, current_cds

def make_plot(pred_src, curr_src): # (Predictive, Current)
    ## Create the plot

    # Setup plot tools
    wheel_zoom_tool = WheelZoomTool(maintain_focus=False)
    pan_tool = PanTool()
    hover_tool = HoverTool(
        tooltips=[
            ('Data', '$name'),
            ('Date', '$x{%a %b, %Y}'),
            ('Time', '$x{%R}'),
            ('Value', '$y'),
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
        align='center',
        sizing_mode='stretch_both',
        toolbar_location = None,
        x_axis_label = None,
        y_axis_label = "Power (MWe)",
        output_backend='webgl',
        )

    # Set action to reset plot
    plot.js_on_event(DoubleTap, CustomJS(args=dict(p=plot), 
    code="""
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
    current_time_line = Span(
        location=current_datetime,
        dimension='height',
        line_color='white',
        line_dash='dashed',
        line_width=2
    )
    plot.add_layout(current_time_line)
    i = -1
    for data_label, data_column in PLOT_LABELS_FOR_DATA_COLS.items():
        if 'Timestamp' in data_label:
            continue
        i += 1
        if 'Field' in data_label:
            y_range_name = 'mwt'
            level = 'underlay'
            line_width = 2
        else:
            y_range_name = 'default'
            level = 'glyph' if 'Actual' in data_label else 'underlay'
            line_width = 3 if 'Actual' in data_label else 2
        color = cc.glasbey_cool[i]
        active_labels = [plot_select.labels[i] for i in plot_select.active]
        PLOT_LINES[data_label] = plot.line(
            x = PLOT_LABELS_FOR_DATA_COLS['Timestamp'],
            y = data_column,
            line_color = color,
            line_alpha = 0.7,
            hover_line_color = color,
            hover_alpha = 1.0,
            source = curr_src if PLOT_LABELS_FOR_DATA_COLS[data_label] in curr_src.column_names else pred_src,
            name = data_label,
            visible = _strip_unit(data_label) in active_labels,
            y_range_name = y_range_name,
            level = level,
            line_width = line_width,
        )

        if 'Field' in data_label:
            plot.extra_y_ranges['mwt'].renderers.append(PLOT_LINES[data_label])
        else:
            plot.y_range.renderers.append(PLOT_LINES[data_label])
        legend_item = LegendItem(label=data_label, renderers=[PLOT_LINES[data_label]])
        legend.items.append(legend_item)
    # styling
    plot = butils.style(plot)
    plot.add_layout(legend, 'below')

    return plot

@gen.coroutine
def _periodic_callback():
    ## Do a live update on the minute
    global current_datetime
    new_current_datetime = datetime.datetime.now().replace(second=0, microsecond=0)
    q = queue.Queue()
    # Update timeline for current time
    getattr(plot, 'center')[2].location = new_current_datetime
    # Current Data
    current_data_df = _getDashboardData(
        q,
        current_datetime,
        new_current_datetime,
        CURRENT_DATA_COLS,
    )
    curr_src.stream(current_data_df)
    df_temp = curr_src.to_df().drop([0]).drop('index', axis=1)
    curr_src.data.update(ColumnDataSource(df_temp).data)
    # Future Data
    predictive_data_df = _getDashboardData(
        q,
        current_datetime,
        new_current_datetime,
        FUTURE_DATA_COLS,  # the last parameter isn't really the 'scheduled'
    )
    pred_src.stream(predictive_data_df)
    df_temp = pred_src.to_df().drop([0]).drop('index', 1)
    pred_src.data.update(ColumnDataSource(df_temp).data)
    current_datetime = new_current_datetime
    return

## Create widget layout
# Create radio button group widget
time_window = RadioButtonGroup(
    labels=list(TIME_BOXES.keys()),
    active=0,
    width_policy='min',
    height=31)

def _time_window_callback(attr, _, new_index):
    assert('active' == attr)
    time_box = list(TIME_BOXES.keys())[new_index]
    [new_pred_src, new_curr_src] = make_dataset(time_box)
    pred_src.data.update(new_pred_src.data)
    curr_src.data.update(new_curr_src.data)
    return

time_window.on_change('active', _time_window_callback)

# Create Checkbox Select Group Widget
plot_select = CheckboxButtonGroup(
    labels = [
        _strip_unit(label)
        for label in PLOT_LABELS_FOR_DATA_COLS.keys() if label != 'Timestamp'
    ],
    active = [0],
    width_policy='min',
    height=31
)

def _plot_select_callback(attr, _, new_indices):
    assert('active' == attr)
    i = 0
    for label in PLOT_LINES.keys():
        PLOT_LINES[label].visible = i in new_indices
        i += 1
    return

plot_select.on_change('active', _plot_select_callback)

[pred_src, curr_src] = make_dataset('Today')
plot = make_plot(pred_src, curr_src)

widgets = row(
    time_window,
    Spacer(width_policy='max'),
    plot_select)

layout = column(
    widgets,
    Spacer(height=20),
    plot,
    sizing_mode='stretch_width',
    width_policy='max')

curdoc().add_root(layout)
curdoc().add_periodic_callback(_periodic_callback, 60000)
curdoc().title = "Dashboard"
curdoc().theme = Theme(json=theme.json)

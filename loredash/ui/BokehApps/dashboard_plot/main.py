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
from threading import Thread
import queue

TIME_BOXES = {
    'TODAY': 1,
    'LAST_6_HOURS': 6,
    'LAST_12_HOURS': 12,
    'LAST_24_HOURS': 24,
    'LAST_48_HOURS': 48
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

def make_dataset(time_box):
    # Prepare data
    start_date = current_datetime.date()
    end_date = current_datetime
    pred_end_date = current_datetime.date() + datetime.timedelta(days=1)

    if time_box != 'TODAY':
        start_date = current_datetime - datetime.timedelta(hours=TIME_BOXES[time_box])
        pred_end_date = current_datetime

    q = queue.Queue()

    # Current Data
    thread = Thread(target=getDashboardData, 
        args=((start_date, end_date), 
            [data_columns[i] for i in CURRENT_DATA_COLS],
            q))
    thread.start()
    thread.join()
    current_data_df = q.get()
    current_cds = ColumnDataSource(current_data_df)

    # Future Data
    thread = Thread(target=getDashboardData, 
        args=((start_date, pred_end_date), 
        [data_columns[i] for i in FUTURE_DATA_COLS],
        q))
    thread.start()
    thread.join()
    predictive_data_df = q.get()
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

    for data_label, data_column in PLOT_LABELS_FOR_DATA_COLS.items():
        if 'Timestamp' in data_label: continue
        data_label_no_unit = re.sub(' \[.*\]$', '', data_label)
        if 'Field' in data_label:
            y_range_name = 'mwt'
            level = 'underlay'
            line_width = 2
        else:
            y_range_name = 'default'
            level = 'glyph' if 'Actual' in data_label else 'underlay'
            line_width = 3 if 'Actual' in data_label else 2

        lines[data_label] = plot.line( 
            x = PLOT_LABELS_FOR_DATA_COLS['Timestamp'],
            y = data_column,
            line_color = label_colors[data_label + '_color'],
            line_alpha = 0.7,
            hover_line_color = label_colors[data_label + '_color'],
            hover_alpha = 1.0,
            source = curr_src if PLOT_LABELS_FOR_DATA_COLS[data_label] in curr_src.column_names else pred_src,
            name = data_label,
            visible = data_label_no_unit in [plot_select.labels[i] for i in plot_select.active],
            y_range_name = y_range_name,
            level = level,
            line_width = line_width,
        )

        if 'Field' in data_label:
            plot.extra_y_ranges['mwt'].renderers.append(lines[data_label])
        else:
            plot.y_range.renderers.append(lines[data_label])
        legend_item = LegendItem(label=data_label, renderers=[lines[data_label]])
        legend.items.append(legend_item)

    # styling
    plot = butils.style(plot)
    plot.add_layout(legend, 'below')

    return plot

def update_lines(attr, old, new):
    # Update visible lines
    selected_labels = [plot_select.labels[i] for i in plot_select.active]

    for label in lines.keys():
        label_name_no_units = re.sub(' \[.*\]$', '', butils.col_to_title(label))
        lines[label].visible = label_name_no_units in selected_labels


def update_points(attr, old, new):
    # Get updated time block information
    time_box = list(TIME_BOXES.keys())[time_window.active]

    # Update data
    [new_pred_src, new_curr_src] = make_dataset(time_box)
    pred_src.data.update(new_pred_src.data)
    curr_src.data.update(new_curr_src.data)

@gen.coroutine
def live_update():
    ## Do a live update on the minute
    global current_datetime

    new_current_datetime = datetime.datetime.now().replace(second=0, microsecond=0)

    q = queue.Queue()

    # Update timeline for current time
    getattr(plot, 'center')[2].location = new_current_datetime

    # Current Data
    thread = Thread(target=getDashboardData, 
        args=((current_datetime, new_current_datetime), 
            [data_columns[i] for i in CURRENT_DATA_COLS],
            q))
    thread.start()
    thread.join()
    current_data_df = q.get()

    curr_src.stream(current_data_df)
    df_temp = curr_src.to_df().drop([0]).drop('index', axis=1)
    curr_src.data.update(ColumnDataSource(df_temp).data)

    # Future Data
    thread = Thread(target=getDashboardData, 
        args=((current_datetime, new_current_datetime), 
        [data_columns[i] for i in FUTURE_DATA_COLS],                    # the last parameter isn't really the 'scheduled'
        q))
    thread.start()
    thread.join()
    predictive_data_df = q.get()
    
    pred_src.stream(predictive_data_df)
    df_temp = pred_src.to_df().drop([0]).drop('index', 1)
    pred_src.data.update(ColumnDataSource(df_temp).data)

    current_datetime = new_current_datetime


## Create widget layout
# Create radio button group widget
time_window = RadioButtonGroup(
    labels=["Today", "Last 6 Hours", "Last 12 Hours", "Last 24 Hours", "Last 48 Hours"], 
    active=0,
    width_policy='min',
    height=31)
time_window.on_change('active', update_points)

# Create Checkbox Select Group Widget
plot_select = CheckboxButtonGroup(
    labels = [label for label in data_labels_no_units if label != 'Timestamp'],
    active = [0],
    width_policy='min',
    height=31
)

plot_select.on_change('active', update_lines)

# Set initial plot information
initial_plots = [butils.title_to_col(plot_select.labels[i]) for i in plot_select.active]

[pred_src, curr_src] = make_dataset('TODAY')
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
curdoc().add_periodic_callback(live_update, 60000)
curdoc().title = "Dashboard"
curdoc().theme = Theme(json=theme.json)
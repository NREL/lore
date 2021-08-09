# Bokeh
from bokeh.plotting import figure
from bokeh.models import Span, ColumnDataSource, LinearAxis, DataRange1d, Legend, LegendItem, PanTool, WheelZoomTool, HoverTool, CustomJS
from bokeh.models.widgets import CheckboxButtonGroup, RadioButtonGroup
from bokeh.layouts import column, row, Spacer
from bokeh.themes import Theme
from bokeh.io import curdoc
from bokeh.events import DoubleTap

import colorcet
import datetime
import pandas
import queue
import re
import threading
from tornado import gen
import pytz

from mediation.models import TechData
from ui import TIMEZONE_STRING

# theme.py is loredash/io/BokehApps/theme/theme.py. It isn't an external
# package. It's also different from `bokeh.themes`. There is only one constant
# in it.
# bokeh_utils.py is loredash/io/BokehApps/bokeh_utils/bokeh_utils.py.
# These both need the local path.
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import bokeh_utils.bokeh_utils as butils
from theme import theme as _loredash_ui_theme
LOREDASH_UI_THEME = _loredash_ui_theme.json

TIME_BOXES = {
    'Today': datetime.timedelta(hours=1),
    'Last 6 Hours': datetime.timedelta(hours=6),
    'Last 12 Hours': datetime.timedelta(hours=12),
    'Last 24 Hours': datetime.timedelta(hours=24),
    'Last 48 Hours': datetime.timedelta(hours=48),
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

# This is a global variable that will be updated by the live callback. Note that timestamps in db are in UTC.
current_datetime = datetime.datetime.now(datetime.timezone.utc).replace(second=0, microsecond=0)

# A global variable to hold all of the plot lines!
PLOT_LINES = {}

def getDashboardData(queue, date_range, columns):
    """
    Get the dashboard data corresponding `columns` over the `date_range` given
    as the tuple `(date_start, date_stop)`. Store the result in `queue`.
    """
    rows = TechData.objects.filter(timestamp__range=date_range).values_list(*columns)
    df = pandas.DataFrame.from_records(rows)
    if not df.empty:
        df.columns = columns

        # Convert UTC to local time then strip out timezone info as bokeh.ColumnDataSource improperly handles it
        tz = pytz.timezone(TIMEZONE_STRING)
        df['timestamp'] = df['timestamp'].dt.tz_convert(tz).dt.tz_localize(None)
    else:
        df = pandas.DataFrame(columns=columns)

    # Scale values
    if 'W_grid_no_derate' in df.columns:
        df['W_grid_no_derate'] *= 1.e-3             # [MWh]
    if 'W_grid_with_derate' in df.columns:
        df['W_grid_with_derate'] *= 1.e-3           # [MWh]
    if 'Q_tower_incident' in df.columns:
        df['Q_tower_incident'] *= 1.e-3             # [MWh]
    if 'Q_field_incident' in df.columns:
        df['Q_field_incident'] *= 1.e-3             # [MWh]

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
    start_date = current_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = current_datetime
    pred_end_date = start_date + datetime.timedelta(days=1)
    if time_box != 'Today':
        start_date = current_datetime - TIME_BOXES[time_box]
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
            ('Date', '$x{%a %b %d, %Y}'),
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
    plot.js_on_event(
        DoubleTap,
        CustomJS(args=dict(p=plot), code="p.reset.emit()"),
    )

    plot.toolbar.active_drag = pan_tool
    plot.toolbar.active_scroll = wheel_zoom_tool   

    plot.x_range.range_padding=0.005
    plot.x_range.range_padding_units="percent"

    plot.extra_y_ranges = {"mwt": DataRange1d()}
    plot.add_layout(LinearAxis(y_range_name="mwt", axis_label="Power (MWt)"), 'right')
    
    legend = Legend(orientation='horizontal', location='top_center', spacing=10)
    
    # Add current time vertical line
    tz = pytz.timezone(TIMEZONE_STRING)
    current_datetime_local = current_datetime.astimezone(tz).replace(tzinfo=None)
    plot.add_layout(
        Span(
            location=current_datetime_local,
            dimension='height',
            line_color='white',
            line_dash='dashed',
            line_width=2
        )
    )
    i = -1
    for label, data in PLOT_LABELS_FOR_DATA_COLS.items():
        if 'Timestamp' in label:
            continue
        i += 1
        color = colorcet.glasbey_cool[i]
        active_labels = [plot_select.labels[i] for i in plot_select.active]
        if PLOT_LABELS_FOR_DATA_COLS[label] in curr_src.column_names:
            source = curr_src
        else:
            source = pred_src
        PLOT_LINES[label] = plot.line(
            x = PLOT_LABELS_FOR_DATA_COLS['Timestamp'],
            y = data,
            line_color = color,
            line_alpha = 0.7,
            hover_line_color = color,
            hover_alpha = 1.0,
            source = source,
            name = label,
            visible = _strip_unit(label) in active_labels,
            y_range_name = 'mwt' if 'Field' in label else 'default',
            level = 'glyph' if 'Actual' in label else 'underlay',
            line_width = 3 if 'Actual' in label else 2,
        )
        if 'Field' in label:
            plot.extra_y_ranges['mwt'].renderers.append(PLOT_LINES[label])
        else:
            plot.y_range.renderers.append(PLOT_LINES[label])
        legend.items.append(
            LegendItem(label=label, renderers=[PLOT_LINES[label]])
        )
    # styling
    plot = butils.style(plot)
    plot.add_layout(legend, 'below')
    return plot

@gen.coroutine
def _periodic_callback():
    ## Do a live update on the minute. Note that timestamps in db are in UTC.
    global current_datetime
    new_current_datetime = datetime.datetime.now(datetime.timezone.utc).replace(second=0, microsecond=0)
    q = queue.Queue()
    # Update timeline for current time
    getattr(plot, 'center')[2].location = new_current_datetime      # update position of vertical dotted line to current time
    # Current Data
    current_data_df = _getDashboardData(
        q,
        current_datetime,
        new_current_datetime,
        CURRENT_DATA_COLS,
    )
    if not current_data_df.empty:       # updating the columndatasource with an empty dataframe will change the timestamp column data format to an integer
        print(str(current_data_df['timestamp'].values[-1]))
        curr_src.stream(current_data_df)                            # add new data to plot
        df_temp = curr_src.to_df().drop([0]).drop('index', axis=1)  # drop first row (oldest data) then drop index column
        curr_src.data.update(ColumnDataSource(df_temp).data)        # create new columndata object, extract just the data, then update the original columndata source of the data...why?
    # Future Data
    predictive_data_df = _getDashboardData(
        q,
        current_datetime,
        new_current_datetime,
        FUTURE_DATA_COLS,  # the last parameter isn't really the 'scheduled'
    )
    if not predictive_data_df.empty:
        pred_src.stream(predictive_data_df)
        df_temp = pred_src.to_df().drop([0]).drop('index', 1)
        pred_src.data.update(ColumnDataSource(df_temp).data)
    current_datetime = new_current_datetime
    return

###
### Make the widget to control the time windows.
###

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

###
### Make the widget to control which lines are shown.
###

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

###
### Make the actual plot.
###

[pred_src, curr_src] = make_dataset('Today')
plot = make_plot(pred_src, curr_src)

doc = curdoc()
doc.add_root(
    column(
        row(
            time_window,
            Spacer(width_policy='max'),
            plot_select,
        ),
        Spacer(height=20),
        plot,
        sizing_mode='stretch_width',
        width_policy='max',
    )
)
doc.add_periodic_callback(_periodic_callback, 60000)
doc.title = "Dashboard"
doc.theme = Theme(json=LOREDASH_UI_THEME)

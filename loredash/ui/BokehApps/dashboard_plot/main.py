import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from bokeh import events as bokeh_events
from bokeh import io as bokeh_io
from bokeh import layouts as bokeh_layouts
from bokeh import models as bokeh_models
from bokeh import plotting as bokeh_plotting
from bokeh import themes as bokeh_themes

import colorcet
import datetime
import pandas
import queue
import threading

from mediation.models import TechData

# bokeh_utils.py is loredash/io/BokehApps/bokeh_utils/bokeh_utils.py. It isn't
# an external package.
from bokeh_utils import bokeh_utils

# theme.py is loredash/io/BokehApps/theme/theme.py. It isn't an external
# package. It's also different from `bokeh.themes`. There is only one constant
# in it.
from theme import theme as _loredash_ui_theme
LOREDASH_UI_THEME = _loredash_ui_theme.json

TIME_BOXES = {
    'Today': 1,
    'Last 6 Hours': 6,
    'Last 12 Hours': 12,
    'Last 24 Hours': 24,
    'Last 48 Hours': 48,
}

# From the TechData model table
PLOT_LABELS_FOR_DATA_COLS = {
    'Timestamp': 'timestamp',
    'TES [kWht]': 'E_tes_charged',
    'Actual [MWe]': 'W_grid_no_derate',
    'Optimal [MWe]': 'W_grid_with_derate',
    'Field Op. Generated [MWt]': 'Q_tower_incident',
    'Field Op. Available [MWt]': 'Q_field_incident',
}

CURRENT_DATA_COLS = [
    'timestamp',
    'Q_tower_incident',
    'Q_field_incident',
    'E_tes_charged',
]
FUTURE_DATA_COLS = [
    'timestamp',
    'W_grid_with_derate',
    'W_grid_no_derate',
]

current_datetime = datetime.datetime.now().replace(second=0, microsecond=0)

# A global variable for the plot callbacks.
PLOT_LINES = {}

def getDashboardData(queue, date_range, columns):
    """
    Get the dashboard data corresponding `columns` over the `date_range` given
    as the tuple `(date_start, date_stop)`. Store the result in `queue`.
    """
    query = TechData.objects.filter(timestamp__range=date_range)
    df = pandas.DataFrame.from_records(query.values_list(*columns))
    # TODO(odow): what is this if-statement doing?
    if not df.empty:
        df.columns = columns
    else:
        df = pandas.DataFrame(columns=columns)
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
    df = queue.get()
    return bokeh_models.ColumnDataSource(df)

def makeDataset(time_box):
    """
    Prepares the data for plotting. `time_box` must be one of the keys to
    `TIME_BOXES`.
    """
    if time_box == 'Today':
        start_date = current_datetime.date()
        pred_end_date = start_date + datetime.timedelta(days=1)
    else:
        start_date = current_datetime - datetime.timedelta(hours=TIME_BOXES[time_box])
        pred_end_date = current_datetime
    q = queue.Queue()
    predictive_cds = _getDashboardData(
        q,
        start_date,
        pred_end_date,
        FUTURE_DATA_COLS,
    )
    current_cds = _getDashboardData(
        q,
        start_date,
        current_datetime,
        CURRENT_DATA_COLS,
    )
    return predictive_cds, current_cds

def makePlot(predictive_cds, current_cds):
    hover_tool = bokeh_models.HoverTool(
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
    wheel_zoom_tool = bokeh_models.WheelZoomTool(maintain_focus=False)
    pan_tool = bokeh_models.PanTool()
    plot = bokeh_plotting.figure(
        tools=[hover_tool,wheel_zoom_tool, pan_tool],
        x_axis_type="datetime",
        width=650,
        height=525,
        y_axis_label = "Power (MWe)",
        output_backend='webgl',
        sizing_mode='stretch_both',
    )
    # Set action to reset plot
    plot.js_on_event(
        bokeh_events.DoubleTap,
        bokeh_models.CustomJS(args=dict(p=plot), code="p.reset.emit()"),
    )
    plot.toolbar.active_drag = pan_tool
    plot.toolbar.active_scroll = wheel_zoom_tool   
    plot.x_range.range_padding = 0.005
    plot.x_range.range_padding_units = "percent"
    plot.extra_y_ranges = {"mwt": bokeh_models.DataRange1d()}
    plot.add_layout(
        bokeh_models.LinearAxis(y_range_name="mwt", axis_label="Power (MWt)"),
        'right',
    )
    legend = bokeh_models.Legend(
        orientation='horizontal',
        location='top_center',
        spacing=10,
    )
    # Add current time vertical line
    plot.add_layout(
        bokeh_models.Span(
            location=current_datetime,
            dimension='height',
            line_color='white',
            line_dash='dashed',
            line_width=2
        )
    )
    active_plots = [plot_select.labels[i] for i in plot_select.active]
    i = 0
    for label, data in PLOT_LABELS_FOR_DATA_COLS.items():
        if label == 'Timestamp':
            continue
        i += 1
        color = colorcet.glasbey_cool[i]
        PLOT_LINES[label] = plot.line(
            x = PLOT_LABELS_FOR_DATA_COLS['Timestamp'],
            y = data,
            line_color = color,
            line_alpha = 0.7,
            hover_line_color = color,
            hover_alpha = 1.0,
            source = curr_src if PLOT_LABELS_FOR_DATA_COLS[label] in curr_src.column_names else pred_src,
            name = label,
            visible = label in active_plots,
            y_range_name = 'mwt' if 'Field' in label else 'default',
            level = 'glyph' if 'Actual' in label else 'underlay',
            line_width = 3 if 'Actual' in label else 2,
        )
        if 'Field' in label:
            plot.extra_y_ranges['mwt'].renderers.append(PLOT_LINES[label])
        else:
            plot.y_range.renderers.append(PLOT_LINES[label])
        legend.items.append(
            bokeh_models.LegendItem(
                label=label,
                renderers=[PLOT_LINES[label]],
            )
        )
    plot.add_layout(legend, 'below')
    return plot

# These are global variables that get updated in the callbacks!
[pred_src, curr_src] = makeDataset('Last 48 Hours')

# Create radio button group widget
time_window = bokeh_models.RadioButtonGroup(
    labels=list(TIME_BOXES.keys()),
    active=4,
    width_policy='max',
    height=31,
)
def _timeWindowCallback(attr, old, new):
    time_box = list(TIME_BOXES.keys())[time_window.active]
    [new_pred_src, new_curr_src] = makeDataset(time_box)
    pred_src.data.update(new_pred_src.data)
    curr_src.data.update(new_curr_src.data)
    return
time_window.on_change('active', _timeWindowCallback)

# Create Checkbox Select Group Widget
plot_select = bokeh_models.CheckboxButtonGroup(
    labels = [
        label for label in PLOT_LABELS_FOR_DATA_COLS if label != 'Timestamp'
    ],
    active = [i for i in range(len(PLOT_LABELS_FOR_DATA_COLS)-1)],
    width_policy='max',
    height=31,
)
def _plotSelectCallback(attr, old, new):
    active_plots = [plot_select.labels[i] for i in plot_select.active]
    for label in PLOT_LINES.keys():
        PLOT_LINES[label].visible = label in active_plots
    return
plot_select.on_change('active', _plotSelectCallback)

layout = bokeh_layouts.column(
    time_window,
    plot_select,
    bokeh_layouts.Spacer(height=20),
    makePlot(pred_src, curr_src),
    sizing_mode='stretch_width',
    width_policy='max',
)

current_doc = bokeh_io.curdoc()
current_doc.title = "Dashboard"
current_doc.theme = bokeh_themes.Theme(json = LOREDASH_UI_THEME)
current_doc.add_root(layout)

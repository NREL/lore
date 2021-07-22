import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import colorcet
import datetime
import pandas
import pytz
import queue 
import threading
from bokeh import events as bokeh_events
from bokeh import io as bokeh_io
from bokeh import layouts as bokeh_layouts
from bokeh import models as bokeh_models
from bokeh import plotting as bokeh_plotting
from bokeh import themes as bokeh_themes

from mediation.models import SolarForecastData

# theme.py is loredash/io/BokehApps/theme/theme.py. It isn't an external
# package. It's also different from `bokeh.themes`. There is only one constant
# in it.
from theme import theme as _loredash_ui_theme
LOREDASH_UI_THEME = _loredash_ui_theme.json

def latestData(queue, resolution = datetime.timedelta(hours=1)):
    datetime_start = datetime.datetime.now(pytz.UTC)
    query = SolarForecastData.objects.filter(
        timestamp__gte = datetime_start,
    )
    raw_data = pandas.DataFrame(query.values())
    raw_data.set_index('timestamp', inplace=True)
    data = raw_data.resample(resolution).mean()
    # Clean up the ratio by imputing any NaNs that arose to the nearest
    # non-NaN value. This means we assume that the start of the day acts
    # like the earliest observation, and the end of the day looks like the
    # last observation.
    data.interpolate(method = 'linear', inplace = True)
    # However, nearest only works when there are non-NaN values either side.
    # For the first and last NaNs, use bfill and ffill:
    data.fillna(method = 'bfill', inplace = True)
    data.fillna(method = 'ffill', inplace = True)
    if datetime_start is None:
        datetime_start = data.index[0]
    datetime_end = datetime_start + datetime.timedelta(hours=48)
    data = data[data.index < datetime_end + resolution]
    data = data[data.index > datetime_start - resolution]
    queue.put(data)
    return

def makeDataset():
    q = queue.Queue()
    thread = threading.Thread(target = latestData, args = (q,))
    thread.start()
    thread.join()
    data = q.get()
    return bokeh_models.ColumnDataSource(data = data)

def makePlot():
    source = makeDataset()
    hover_tool = bokeh_models.HoverTool(
        line_policy='nearest',
        tooltips=[
            ('Data','$name'),
            ('Date', '$x{%a %b, %Y}'),
            ('Time (UTC)', '$x{%R}'),
            ('Value', '$y W/m^2')
        ],
        formatters={
            '$x':'datetime'
        }
    )
    wheel_zoom_tool = bokeh_models.WheelZoomTool(maintain_focus=False)
    pan_tool = bokeh_models.PanTool()
    plot = bokeh_plotting.figure(
        tools = [hover_tool, wheel_zoom_tool, pan_tool],
        x_axis_type = 'datetime',
        width = 650,
        height = 525,
        y_axis_label = 'Power [W/m^2]',
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
    # Hide the toolbar
    plot.toolbar_location = None
    clear_sky = plot.line(
        x = 'timestamp',
        y = 'clear_sky',
        source = source,
        name = 'Clear sky',
        line_color = 'white',
        line_dash = 'dotted',
        line_width = 2,
    )
    best_guess = plot.line(
        x = 'timestamp',
        y = 'dni_50',
        source = source,
        name = 'Best guess',
        line_color = colorcet.fire[200],
        line_width = 3,
    )
    legend_items = [
        ('Clear sky', [clear_sky]),
        ('Best guess', [best_guess])
    ]
    for (c, name, l, u) in [
        (100, '80% chance', 'dni_10', 'dni_90'),
        (150, '50% chance', 'dni_25', 'dni_75')
    ]:
        varea = plot.varea(
            x = 'timestamp',
            y1 = l,
            y2 = u,
            name = name,
            source = source,
            level = 'underlay',
            fill_alpha = 0.4,
            fill_color = colorcet.fire[c],
        )
        legend_items.append((name, [varea]))
    legend = bokeh_models.Legend(
        items = legend_items,
        orientation = 'horizontal',
        location = 'top_left',
        spacing = 10,
    )
    plot.add_layout(legend, 'below')
    return plot

plot = makePlot()

layout = bokeh_layouts.column(
    bokeh_models.widgets.Div(text="""<h3>Solar Forecast</h3>"""),
    plot,
    sizing_mode = 'stretch_width',
    width_policy = 'max',
)

current_doc = bokeh_io.curdoc()
current_doc.title = "Solar Forecast Plot"

current_doc.theme = bokeh_themes.Theme(json = LOREDASH_UI_THEME)
current_doc.add_root(layout)

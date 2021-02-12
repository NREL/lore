from bokeh import io as bokeh_io
from bokeh import layouts as bokeh_layouts
from bokeh import models as bokeh_models
from bokeh import plotting as bokeh_plotting
from bokeh import themes as bokeh_themes

import colorcet

from mediation import forecasts
from mediation import models

import queue 
import threading

# theme.py is loredash/io/BokehApps/theme/theme.py. It isn't an external
# package. It's also different from `bokeh.themes`. There is only one constant
# in it.
from theme import theme as _loredash_ui_theme
LOREDASH_UI_THEME = _loredash_ui_theme.json

def latestData(queue):
    plant = models.PlantConfig.objects.get(pk = 1)
    forecaster = forecasts.SolarForecast(
        plant.latitude,
        plant.longitude,
        plant.timezone_string,
        plant.elevation,
    )
    data = forecaster.latestForecast().reset_index()
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
            ('Time', '$x{%R}'),
            ('Value', '$y W/m^2')
        ],
        formatters={
            '$x':'datetime'
        }
    )
    plot = bokeh_plotting.figure(
        tools = [hover_tool],
        x_axis_type = 'datetime',
        width = 650,
        height = 525,
        y_axis_label = 'Power [W/m^2]',
    )
    clear_sky = plot.line(
        x = 'forecast_for',
        y = 'clear_sky',
        source = source,
        name = 'Clear sky',
        line_color = 'white',
        line_dash = 'dashed',
        line_width = 3,
    )
    best_guess = plot.line(
        x = 'forecast_for',
        y = '0.5',
        source = source,
        name = 'Best guess',
        line_color = colorcet.fire[200],
        line_width = 5,
    )
    legend_items = [
        ('Clear sky', [clear_sky]),
        ('Best guess', [best_guess])
    ]
    for (c, name, l, u) in [
        (100, '80% chance', '0.1', '0.9'),
        (150, '50% chance', '0.25', '0.75')
    ]:
        varea = plot.varea(
            x = 'forecast_for',
            y1 = l,
            y2 = u,
            name = name,
            source = source,
            level = 'underlay',
            fill_alpha = 0.6,
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

from bokeh import io as bokeh_io
from bokeh import layouts
from bokeh import models as bokeh_models
from bokeh import plotting
from bokeh import themes
from mediation import forecasts
from mediation import models
import queue 
from theme import theme
from threading import Thread

def latest_data(queue):
    plant = models.PlantConfig.objects.get(pk = 1)
    forecaster = forecasts.SolarForecast(
        plant.latitude,
        plant.longitude,
        plant.timezone_string,
        plant.elevation,
    )
    data = forecaster.latest_forecast().reset_index()
    queue.put(data)
    return

def make_dataset():
    q = queue.Queue()
    thread = Thread(target = latest_data, args = (q,))
    thread.start()
    thread.join()
    data = q.get()
    return bokeh_models.ColumnDataSource(data = data)

def make_plot():
    source = make_dataset()
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
    plot = plotting.figure(
        tools = [hover_tool],
        x_axis_type = 'datetime',
        width = 650,
        height = 525,
        y_axis_label = 'Power [W/m^2]',
    )
    legend = bokeh_models.Legend(
        orientation = 'horizontal', 
        location = 'top_left', 
        spacing = 10,
    )
    clear_sky = plot.line(
        x = 'forecast_for',
        y = 'clear_sky',
        source = source,
        name = 'Clear sky',
        line_width = 2,
        line_dash = 'dashed',
    )
    legend.items.append(
        bokeh_models.LegendItem(label = 'Clear sky', renderers = [clear_sky])
    )
    best_guess = plot.line(
        x = 'forecast_for',
        y = '0.5',
        source = source,
        name = 'Best guess',
        line_width = 5,
    )
    legend.items.append(
        bokeh_models.LegendItem(label = 'Best guess', renderers = [best_guess])
    )

    for (l, u) in [('0.1', '0.9'), ('0.25', '0.75')]:
        band = bokeh_models.Band(
            base = 'forecast_for',
            lower = l,
            upper = u,
            source = source,
            level = 'underlay',
            fill_alpha = 0.2,
        )
        plot.add_layout(band)
    plot.add_layout(legend, 'below')
    return plot

plot = make_plot()

layout = layouts.column(
    bokeh_models.widgets.Div(text="""<h3>Solar Forecast</h3>"""),
    plot,
    sizing_mode = 'stretch_width',
    width_policy = 'max',
)

current_doc = bokeh_io.curdoc()
current_doc.title = "Solar Forecast Plot"
current_doc.theme = themes.Theme(json=theme.json)
current_doc.add_root(layout)

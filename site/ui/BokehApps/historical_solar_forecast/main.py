# Bokeh
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LinearAxis, DataRange1d, Legend, LegendItem, Band, Range1d, CustomJS, HoverTool, WheelZoomTool, PanTool, Span
from bokeh.models.widgets import CheckboxButtonGroup, RadioButtonGroup, Div, DateSlider, Slider, Button, Select, DatePicker
from bokeh.palettes import Category20
from bokeh.layouts import column, row, WidgetBox, Spacer
from bokeh.themes import built_in_themes
from bokeh.events import DoubleTap
from bokeh.io import curdoc

# Data manipulation
import pandas as pd
import datetime
import numpy as np
import re
from scipy.signal import savgol_filter

# Asyncronous Access to Django DB
from ui.models import ForecastsSolarData as fsd
from threading import Thread
import queue

data_labels_forecast_solar = list(map(lambda col: col.name, fsd._meta.get_fields()))
current_datetime = datetime.datetime.now().replace(year=2010)

plus_minus_regx = re.compile('.*(?<!_minus)(?<!_plus)$')
base_data_labels = list(filter(plus_minus_regx.search, data_labels_forecast_solar))
num_colors = len(base_data_labels[2:])
label_colors = {col+'_color': i*2 for i,col in enumerate(base_data_labels[2:])}
lines = {}
bands = {}

def getForecastSolarData(_range, queue):
    queryset = fsd.objects.filter(timestamp__range=_range).values_list(*(data_labels_forecast_solar[1:]))
    df = pd.DataFrame.from_records(queryset)
    df.columns = data_labels_forecast_solar[1:]
    queue.put(df)

def getTimeRange(queue):
    times = fsd.objects.values_list('timestamp')
    
    start_date = times.order_by('timestamp').first()[0]
    end_date = current_datetime
    queue.put((start_date, end_date))

def make_dataset(start_date, end_date, distribution):
    # Prepare data
    if end_date > current_datetime:
        return src
    
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
        tools=[wheel_zoom_tool, pan_tool, hover_tool], # this gives us our tools
        x_axis_type="datetime",
        toolbar_location = None,
        x_axis_label = None,
        y_axis_label = "Power (W/m^2)",
        width_policy='max',
        height_policy='max',
        output_backend='webgl'
        )

    # Set action to reset plot
    plot.js_on_event(DoubleTap, CustomJS(args=dict(p=plot), code="""
        p.reset.emit()
    """))

    plot.toolbar.active_drag = pan_tool
    plot.toolbar.active_scroll = wheel_zoom_tool

    plot.x_range.range_padding=0.02
    plot.x_range.range_padding_units="percent"

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

    for label in base_data_labels[2:]:

        legend_label = col_to_title_upper(label)
        lower_upper_regex = re.compile(label+'(_plus|_minus)')
        if len(list(filter(lower_upper_regex.search, data_labels_forecast_solar))):
            bands[label] = Band(
                base='timestamp',
                lower= label + '_lower',
                upper= label + '_upper',
                source=src,
                level = 'underlay',
                fill_alpha=0.4,
                fill_color=Category20[20][label_colors[label+'_color']+1],
                line_width=1, 
                line_color='black',
                visible = label in [title_to_col(plot_select.labels[i]) for i in plot_select.active],
                name = label,
                )
            plot.add_layout(bands[label])
            lines[label] = plot.line( 
                x='timestamp',
                y=label,
                line_color = Category20[20][label_colors[label+'_color']], 
                line_alpha = 1.0,
                line_width=3,
                source=src,
                visible = label in [title_to_col(plot_select.labels[i]) for i in plot_select.active],
                name = legend_label,
               
                )
            legend_item = LegendItem(label=legend_label, renderers=[lines[label]])
            legend.items.append(legend_item)
        else:
            color = Category20[20][label_colors[label+'_color']]
            lines[label] = plot.line( 
                x='timestamp',
                y=label,
                line_color = color, 
                line_alpha = 1.0,
                line_width=3,
                source=src,
                visible = label in [title_to_col(plot_select.labels[i]) for i in plot_select.active],
                name = legend_label,
               
                )
            legend_item = LegendItem(label=legend_label, renderers=[lines[label]])
            legend.items.append(legend_item)

    # styling
    plot = style(plot)

    plot.add_layout(legend, 'below')

    return plot

def col_to_title_upper(label):
    # Convert column name to title

    legend_label = ' '.join([word.upper() for word in label.split('_')])

    return legend_label

def title_to_col(title):
    # Convert title to a column name

    col_name = title.lower().replace(' ','_')
    return col_name

def update_points(attr, old, new):
    # Update range when sliders move and update button is clicked
    delta = datetime.timedelta(hours=date_span_slider.value)
    selected_date =date_slider.value_as_datetime
    range_start = range_end = selected_date
    if( datetime.timedelta(0) > delta):
        range_start += delta
    else:
        range_end += delta

    new_src = make_dataset(range_start, range_end, distribution_select.value)
    src.data.update(new_src.data)


def update_lines(attr, old, new):
    # Update visible lines 
    selected_labels = [plot_select.labels[i] for i in plot_select.active]

    for label in lines.keys():
        label_name = col_to_title_upper(label)
        lines[label].visible = label_name in selected_labels
        if label in bands.keys():
            bands[label].visible = lines[label].visible
        



## Create widgets
# Select for plots to show
plot_select = CheckboxButtonGroup(
    labels = list(map(col_to_title_upper, base_data_labels[2:])),
    active = [0],
    width_policy='min',
    css_classes=['bokeh_buttons']
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
    start=start_date.date(), 
    end=end_date, 
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

distribution_init = 'Discrete'
distribution_select = Select(
    value=distribution_init,
    options=['Discrete', 'Smoothed'],
    width=125)
distribution_select.on_change('value', update_points)

title = Div(text="""<h3>Solar</h3>""")

# Set initial plot information
initial_plots = [title_to_col(plot_select.labels[i]) for i in plot_select.active]

delta_init = datetime.timedelta(hours=24)
src = make_dataset(current_datetime - delta_init, current_datetime, distribution_init)

plot = make_plot(src)

# Setup Widget Layouts
layout = column(
    row(
        title,
        Spacer(width_policy='max'),
        date_slider,
        date_span_slider),
    row(
        Spacer(width_policy='max'),
        distribution_select,
        plot_select
    ), 
    plot, 
    max_height=525,
    height_policy='max', 
    width_policy='max')

# Show to current document/page
curdoc().add_root(layout)
curdoc().theme = 'dark_minimal'
curdoc().title = "Historical Solar Forecast Plot"
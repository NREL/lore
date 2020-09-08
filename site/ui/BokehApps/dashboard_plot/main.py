# Bokeh
from bokeh.plotting import figure
from bokeh.models import Span, ColumnDataSource, LinearAxis, DataRange1d, Legend, LegendItem, PanTool, WheelZoomTool, HoverTool, CustomJS
from bokeh.models.widgets import Button, CheckboxButtonGroup, RadioButtonGroup
from bokeh.palettes import Category20
from bokeh.layouts import column, row, WidgetBox, Spacer
from bokeh.themes import Theme
from bokeh.io import curdoc
from bokeh.events import DoubleTap
from tornado import gen
import sys
sys.path.append('theme')
sys.path.append('bokeh_utils')
import theme
import bokeh_utils as butil

# Data manipulation
import pandas as pd
import datetime
import re

# Asyncronous access to Django DB
from ui.models import DashboardDataRTO as dd
from threading import Thread
import queue

TIME_BOXES = {'TODAY': 1,
              'LAST_6_HOURS': 6,
              'LAST_12_HOURS': 12,
              'LAST_24_HOURS': 24,
              'LAST_48_HOURS': 48
              }

data_labels = list(map(lambda col: col.name, dd._meta.get_fields()))
current_datetime = datetime.datetime.now().replace(year=2010, second=0) # Eventually the year will be removed once live data is added

label_colors = {}
for i, data_label in enumerate(data_labels[2:]):
    label_colors.update({
        data_label: Category20[12][i]
    })
lines = {}

def getDashboardData(_range, _values_list, queue):
    queryset = dd.objects.filter(timestamp__range=_range).values_list(*_values_list)
    df = pd.DataFrame.from_records(queryset)
    df.columns=_values_list
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
            ['timestamp', 'actual', 'field_operation_generated', 'field_operation_available'], 
            q))
    thread.start()
    thread.join()
    current_data_df = q.get()
    current_cds = ColumnDataSource(current_data_df)

    # Future Data
    thread = Thread(target=getDashboardData, 
        args=((start_date, pred_end_date), 
        ['timestamp', 'optimal', 'scheduled'], 
        q))
    thread.start()
    thread.join()
    predictive_data_df = q.get()
    predictive_cds = ColumnDataSource(predictive_data_df)

    return predictive_cds, current_cds

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
        width_policy='max',
        height_policy='max',
        toolbar_location = None,
        x_axis_label = None,
        y_axis_label = "Power (MWe)",
        output_backend='webgl',
        )

    plot.css_classes = ['plot']

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

    for label in data_labels[2:]:
        legend_label = col_to_title(label)
        if 'field' in label:
            lines[label] = plot.line( 
                x='timestamp',
                y=label,
                line_color = label_colors[label], 
                line_alpha = 0.7, 
                hover_line_color = label_colors[label],
                hover_alpha = 1.0,
                y_range_name='mwt',
                level='underlay',
                source = curr_src,
                line_width=2,
                visible=label in [title_to_col(plot_select.labels[i]) for i in plot_select.active],
                name=legend_label
                )

            legend_item = LegendItem(label=legend_label + " [MWt]", renderers=[lines[label]])
            legend.items.append(legend_item)
            plot.extra_y_ranges['mwt'].renderers.append(lines[label])

        else:
            lines[label] = plot.line( 
                x='timestamp',
                y=label,
                line_color = label_colors[label], 
                line_alpha = 0.7, 
                hover_line_color = label_colors[label],
                hover_alpha = 1.0,
                source= curr_src if label == 'actual' else pred_src,
                level='glyph' if label == 'actual' else 'underlay',
                line_width=3 if label == 'actual' else 2,
                visible=label in [title_to_col(plot_select.labels[i]) for i in plot_select.active],
                name=legend_label,
                )

            legend_item = LegendItem(label=legend_label + " [MWe]", renderers=[lines[label]])
            legend.items.append(legend_item)
            plot.y_range.renderers.append(lines[label])

    # styling
    plot = style(plot)

    plot.add_layout(legend, 'below')

    return plot

def col_to_title(label):
    # Convert column name to title

    legend_label = ' '.join([word.title() for word in label.split('_')])

    return legend_label

def title_to_col(title):
    # Convert title to a column name

    col_name = title.lower().replace(' ','_')
    return col_name

def update_lines(attr, old, new):
    # Update visible lines
    selected_labels = [plot_select.labels[i] for i in plot_select.active]

    for label in lines.keys():
        label_name = col_to_title(label)
        lines[label].visible = label_name in selected_labels


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

    new_current_datetime = datetime.datetime.now().replace(year=2010, second=0, microsecond=0) # Until live data is being used

    q = queue.Queue()

    # Update timeline for current time
    getattr(plot, 'center')[2].location = new_current_datetime

    # Current Data
    thread = Thread(target=getDashboardData, 
        args=((current_datetime, new_current_datetime), 
            ['timestamp', 'actual', 'field_operation_generated', 'field_operation_available'], 
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
        ['timestamp', 'optimal', 'scheduled'],
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
    width_policy='min')
time_window.on_change('active', update_points)

# Create Checkbox Select Group Widget
labels_list = [col_to_title(label) for label in data_labels[2:]]
plot_select = CheckboxButtonGroup(
    labels = labels_list,
    active = [0],
    width_policy='min'
)

plot_select.on_change('active', update_lines)

# Set initial plot information
initial_plots = [title_to_col(plot_select.labels[i]) for i in plot_select.active]

[pred_src, curr_src] = make_dataset('TODAY')
plot = make_plot(pred_src, curr_src)

widgets = row(
    time_window,
    Spacer(width_policy='max'),
    plot_select)

layout = column(
    widgets, 
    plot, 
    max_height=525, 
    height_policy='max', 
    width_policy='max')

curdoc().add_root(layout)
curdoc().add_periodic_callback(live_update, 60000)
curdoc().title = "Dashboard"
curdoc().theme = Theme(json=theme.json)
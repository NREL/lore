from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LinearAxis, DataRange1d, Legend, LegendItem, Span
from bokeh.models.widgets import Button, CheckboxButtonGroup, RadioButtonGroup
from bokeh.palettes import Category20
from bokeh.layouts import column, row, WidgetBox, Spacer
from bokeh.themes import built_in_themes

import pandas as pd
from bokeh.io import curdoc
import sqlite3
import datetime
import re

from ui.models import DashboardDataRTO as dd
from asgiref.sync import sync_to_async, async_to_sync
import asyncio
from threading import Thread
import queue

# DEBUG
import pdb

TIME_BOXES = {'TODAY': 1,
              'LAST_6_HOURS': 6,
              'LAST_12_HOURS': 12,
              'LAST_24_HOURS': 24,
              'LAST_48_HOURS': 48
              }

data_labels = list(map(lambda col: col.name, dd._meta.get_fields()))

current_datetime = datetime.datetime.now().replace(year=2010) # Eventually the year will be removed once live data is added
delta_start = datetime.timedelta(days=2)
delta_end = datetime.timedelta(days=1)

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
        end_date = current_datetime
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

    time = pred_src.data['timestamp']
    plot = figure(
        tools="", # this gives us our tools
        x_axis_type="datetime",
        width_policy='max',
        height_policy='max',
        toolbar_location = None,
        x_axis_label = None,
        y_axis_label = "Power (MWe)",
        output_backend='webgl'
        )

    plot.x_range.range_padding=0.02
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
                )

            legend_item = LegendItem(label=legend_label + " [MWe]", renderers=[lines[label]])
            legend.items.append(legend_item)
    
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

def update(attr, old, new):
    # Update plots when widgets change

    # Update visible plots
    for label in lines.keys():
        label_name = col_to_title(label)
        lines[label].visible = label_name in [plot_select.labels[i] for i in plot_select.active]

    # Get updated time block information
    time_box = list(TIME_BOXES.keys())[radio_button_group.active]

    # Update data
    [new_pred_src, new_curr_src] = make_dataset(time_box)
    pred_src.data = dict(new_pred_src.data)
    curr_src.data = dict(new_curr_src.data)

## Create widget layout
# Create radio button group widget
radio_button_group = RadioButtonGroup(
    labels=["Today", "Last 6 Hours", "Last 12 Hours", "Last 24 Hours", "Last 48 Hours"], 
    active=0,
    width_policy='min')
radio_button_group.on_change('active', update)

# Create Checkbox Select Group Widget
labels_list = [col_to_title(label) for label in data_labels[2:]]
plot_select = CheckboxButtonGroup(
    labels = labels_list,
    active = [0],
    width_policy='min'
)

plot_select.on_change('active', update)

# Set initial plot information
initial_plots = [title_to_col(plot_select.labels[i]) for i in plot_select.active]

[pred_src, curr_src] = make_dataset('TODAY')
plot = make_plot(pred_src, curr_src)

widgets = row(
    radio_button_group,
    Spacer(width_policy='max'),
    plot_select)

layout = column(widgets, plot, max_height=525, height_policy='max', width_policy='max')

curdoc().add_root(layout)
curdoc().theme = 'dark_minimal'
curdoc().title = "Dashboard"
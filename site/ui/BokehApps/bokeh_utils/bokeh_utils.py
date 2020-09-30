import datetime

# use global for earliest date
BEGINNING_OF_TIME = datetime.datetime(year=2010, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)

def get_update_range(date_slider, date_span_slider, current_datetime):
    # Filter date ranges sent to the 'get_dataset' methods
    delta = datetime.timedelta(hours=date_span_slider.value)
    selected_date = date_slider.value_as_datetime
    range_start = range_end = selected_date
    if( datetime.timedelta(0) > delta):
        range_start += delta
    else:
        range_end += delta
    
    # If the whole range would return nothing (no date on the range has occured)
    if range_end > current_datetime and range_start >= current_datetime:
        return range_start, range_end # The make_dataset function will just return the current source
    elif range_end > current_datetime: # If only range_end is greated than the current datetime
        range_start = range_start - (range_end - current_datetime)  
        range_end = current_datetime
    elif range_start < BEGINNING_OF_TIME: # Start date is beyond range for dataset
        range_start = BEGINNING_OF_TIME
        range_end = BEGINNING_OF_TIME + abs(delta)

    return range_start, range_end

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

## Converting between title and data column names

def col_to_title(label):
    # Convert column name to title

    legend_label = ' '.join([word.title() for word in label.split('_')])
    legend_label = legend_label.replace('Operation', 'Op.')
    return legend_label

def title_to_col(title):
    # Convert title to a column name

    col_name = title.lower().replace(' ','_')
    return col_name

def col_to_title_upper(label):
    # Convert column name to title

    legend_label = ' '.join([word.upper() for word in label.split('_')])

    return legend_label
import datetime

def get_update_range(date_slider, date_span_slider, current_datetime):
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
    return range_start, range_end
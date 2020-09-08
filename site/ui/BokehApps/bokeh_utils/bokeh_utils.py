import datetime

def get_update_range(date_slider, date_span_slider):
    delta = datetime.timedelta(hours=date_span_slider.value)
    selected_date = datetime.datetime.combine(date_slider.value_as_datetime, datetime.datetime.min.time())
    range_start = range_end = selected_date
    if( datetime.timedelta(0) > delta):
        range_start += delta
    else:
        range_end += delta
    return range_start, range_end
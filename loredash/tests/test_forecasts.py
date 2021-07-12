# Tests for the forecasts.py module. They can be run with:
#  $ python -mpytest

import pytest
import datetime
import pytz

from mediation import forecasts

def test_get_forecast():
    forecaster = forecasts.SolarForecast(38.2, -117.4, -8, 100.0)
    # Choose a start time that is not current, but that NDFD will still have
    # data for.
    datetime_start = datetime.datetime.now(pytz.UTC)
    datetime_start = datetime_start - datetime.timedelta(hours = 24)
    data = forecaster.get_raw_data(
        datetime_start, 
        ['dni', 'dhi', 'temp_air', 'wind_speed', 'ghi'],
    )
    assert(len(data) >= 24)
    assert(datetime_start >= data.index[0])
    assert('dni' in data)
    assert('dhi' in data)
    assert('ghi' in data)
    assert('temp_air' in data)
    assert('wind_speed' in data)
    assert('0.1' in data)
    assert('0.25' in data)
    assert('0.5' in data)
    assert('0.75' in data)
    assert('0.9' in data)
    assert((data['0.9'] >= data['0.1']).all())
    return

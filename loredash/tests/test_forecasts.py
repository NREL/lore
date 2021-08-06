# Tests for the forecasts.py module. They can be run with:
#  $ python -mpytest

import pytest
import datetime
import pytz

from mediation import forecasts

def test_ambient_pressure():
    forecaster = forecasts.SolarForecast(38.2, -117.4, -8, 0.0)
    assert(abs(forecaster.ambient_pressure() - 1013.25) < 1)
    forecaster = forecasts.SolarForecast(38.2, -117.4, -8, 1000.0)
    assert(abs(forecaster.ambient_pressure() - 898.75) < 1)
    return

def test_get_forecast():
    forecaster = forecasts.SolarForecast(38.2, -117.4, -8, 100.0)
    # Choose a start time that is not current, but that NDFD will still have
    # data for.
    datetime_start = datetime.datetime.now(pytz.UTC)
    datetime_start = datetime_start - datetime.timedelta(hours = 24)
    data = forecaster.get_raw_data(datetime_start)
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
    assert('pressure' in data)
    return

def test_openweathermap():
    forecaster = forecasts.SolarForecast(38.2, -117.4, -8, 100.0)
    owm = forecaster.openweathermap.get()
    assert(len(owm) == 48)
    # Check pressure in mbar
    assert(min(owm['pressure']) > 800)
    assert(max(owm['pressure']) < 1200)
    # Check humdity in %
    assert(min(owm['humidity']) >= 0)
    assert(max(owm['humidity']) <= 100)
    return

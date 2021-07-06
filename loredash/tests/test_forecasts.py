# Tests for the forecasts.py module. They can be run with:
#  $ python -mpytest

import pytest

# Our database gets initialized in a weird way because it only happens when you
# run the server. As a work-around, use the current database on disk.
# 
# TODO(odow): remove this setup when we don't need an initialized database.
# TODO: move this file's contents to tests.py
from django.conf import settings
@pytest.fixture(scope='session')
def django_db_setup():
    settings.DATABASES['default'] = {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': 'db.sqlite3',
    }
    return

import datetime
import pandas
from pvlib import location
import pytz

from mediation import forecasts
import rapidjson

# Make sure to mark any tests that need database access.
@pytest.mark.django_db
def test_forecaster_from_plant():
    with open("./config/plant_design.json") as f:
        plant_design = rapidjson.load(f)
    forecaster = forecasts.SolarForecast(
        plant_design['latitude'],
        plant_design['longitude'],
        plant_design['timezone'],
        plant_design['elevation'],
    )
    assert(type(forecaster.plant_location) == location.Location)
    assert(
        type(forecaster.forecast_uncertainty) == forecasts.ForecastUncertainty
    )
    return

@pytest.mark.django_db
def test_latestForecast():
    forecaster = forecasts.SolarForecast(
        38.2,
        -117.4,
        -8,
        100.0,
    )
    data = forecaster.latestForecast()
    assert(len(data) == 50)
    assert('clear_sky' in data.keys())
    assert('0.5' in data.keys())
    # Check clearsky bounds
    assert(min(data['clear_sky']) < 100)
    assert(max(data['clear_sky']) > 800)
    # Check maximum solar in reasonable units, not a ratio
    assert(max(data['1.0']) > 200)
    return

@pytest.mark.django_db
def test_latestForecast_resolution():
    forecaster = forecasts.SolarForecast(
        38.2,
        -117.4,
        -8,
        100.0,
    )
    data = forecaster.latestForecast(resolution = pandas.Timedelta(hours = 2))
    assert(len(data) == 26)
    return

@pytest.mark.django_db
def test_latestForecast_horizon():
    forecaster = forecasts.SolarForecast(
        38.2,
        -117.4,
        -8,
        100.0,
    )
    data = forecaster.latestForecast(horizon = pandas.Timedelta(hours = 24))
    assert(len(data) == 26)
    return

def test_getForecast():
    forecaster = forecasts.SolarForecast(
        38.2,
        -117.4,
        -8,
        100.0,
    )
    # Choose a start time that is not current, but that NDFD will still have
    # data for.
    datetime_start = datetime.datetime.now(pytz.FixedOffset(-480))
    datetime_start = datetime_start - pandas.Timedelta(hours = 24)
    datetime_end = datetime_start + pandas.Timedelta(hours = 24)
    data = forecaster.getForecast(
        datetime_start = datetime_start,
        horizon = pandas.Timedelta(hours = 24),
        resolution = pandas.Timedelta(minutes = 1),
    )
    assert(len(data) >= 24 * 60)
    assert(datetime_start >= data.index[0])
    assert(datetime_end <= data.index[-1])
    assert('dni' in data)
    assert('dhi' in data)
    assert('ghi' in data)
    assert('temp_air' in data)
    assert('wind_speed' in data)
    return

# Tests for the forecasts.py module. They can be run with:
#  $ python -mpytest

import pytest
import datetime
import pytz

from mediation import forecasts
import rapidjson

# Make sure to mark any tests that need database access.
@pytest.mark.django_db
def test_forecaster_from_plant():
    with open("./config/plant_design.json") as f:
        plant_design = rapidjson.load(f, parse_mode=1)
    forecaster = forecasts.SolarForecast(
        plant_design['latitude'],
        plant_design['longitude'],
        plant_design['timezone_string'],
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
        'US/Pacific',
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
        'US/Pacific',
        100.0,
    )
    data = forecaster.latestForecast(resolution = pandas.Timedelta(hours = 2))
    assert(len(data) == 26)
    return

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

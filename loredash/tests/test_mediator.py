# Tests for the mediator.py module.
# To run this file: python -m pytest mediation/test_mediator.py
import pytest

# Our database gets initialized in a weird way because it only happens when you
# run the server. As a work-around, use the current database on disk.
# 
# TODO(odow): remove this setup when we don't need an initialized database.
from django.conf import settings
@pytest.fixture(scope='session')
def django_db_setup():
    settings.DATABASES['default'] = {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': 'db.sqlite3',
    }
    return

import datetime
import pathlib
import pytz

from mediation import mediator

PARENT_DIR = str(pathlib.Path(__file__).parents[1])

def test_tmy3_to_df():
    weather_file = PARENT_DIR + "/data/daggett_ca_34.865371_-116.783023_psmv3_60_tmy.csv"
    datetime_start = datetime.datetime(2021, 1, 1, 0, 0, 0)
    data = mediator.tmy3_to_df(
        weather_file,
        datetime_start,
        datetime_start + datetime.timedelta(days = 7)
    )
    assert(len(data) == 169 == 24 * 7 + 1)
    # Check that the TMY file is in local time!
    assert(sum(data[0:4]['DNI']) == 0)
    # Check that the DNI is in W/m^2
    assert(sum(data['DNI']) == 22216)
    return

@pytest.mark.django_db
def test_get_weather_df():
    plant_design_path = PARENT_DIR + "/../loredash/config/plant_design.json"
    m = mediator.Mediator(
        params = mediator.mediator_params,
        plant_design_path=plant_design_path,
        weather_file = PARENT_DIR + "/data/daggett_ca_34.865371_-116.783023_psmv3_60_tmy.csv",
        update_interval = datetime.timedelta(seconds = 5),
    )
    tzinfo = pytz.FixedOffset(60 * m.plant.design['timezone'])
    datetime_start = datetime.datetime(2021, 1, 1, 0, 0, 0, tzinfo = tzinfo)
    # Test getting one week of weather. Should be pure TMY data.
    weather = m.get_weather_df(
        datetime_start,
        datetime_start + datetime.timedelta(days = 7),
        datetime.timedelta(hours=1),
        m.weather_file,
    )
    assert(len(weather) == 7 * 24)
    # Test getting one day of weather. This should fail because it attempts to
    # get latest forecast, but the time is too old for the NDFD server.
    with pytest.raises(Exception) as err:
        m.get_weather_df(
            datetime_start,
            datetime_start + datetime.timedelta(days = 1),
            datetime.timedelta(hours=1),
            m.weather_file,
            use_forecast = True,
        )
    # Test getting one day of weather with the current forecast.
    datetime_start = datetime.datetime.now(tzinfo)
    tmy_weather = m.get_weather_df(
        datetime_start,
        datetime_start + datetime.timedelta(days = 2),
        datetime.timedelta(hours=0.5),
        m.weather_file,
    )
    # Update database
    m.forecaster.refresh_forecast_in_db(datetime_start)
    forecast_weather = m.get_weather_df(
        datetime_start,
        datetime_start + datetime.timedelta(days = 2),
        datetime.timedelta(hours=0.5),
        m.weather_file,
        use_forecast = True,
    )
    assert(len(tmy_weather) == 2 * 48 - 1)
    for key in ['DNI', 'DHI', 'GHI', 'Temperature', 'Wind Speed']:
        assert(key in tmy_weather)
        assert(key in forecast_weather)
    assert('Clear Sky DNI' in forecast_weather)
    assert(not 'Clear Sky DNI' in tmy_weather)
    assert(len(tmy_weather) == len(forecast_weather))
    assert(sum(tmy_weather['DNI']) != sum(forecast_weather['DNI']))
    assert(sum(tmy_weather['DNI']) > 100)
    assert(not tmy_weather['DNI'].isnull().values.any())
    assert(sum(forecast_weather['DNI']) > 100)
    assert(not forecast_weather['DNI'].isnull().values.any())
    # Check that they are linked up timezone-wise. If the clear-sky is low, the
    # TMY must be too.
    for (tmy, forecast) in zip(tmy_weather['DNI'], forecast_weather['Clear Sky DNI']):
        if forecast < 100:
            assert(tmy <= 200)
    return

def test_normalize_timesteps():
    tzinfo = pytz.FixedOffset(-480)
    datetime_start = datetime.datetime(2021, 1, 1, 1, 32, 0, tzinfo = tzinfo)
    datetime_end = datetime_start + datetime.timedelta(days = 2)
    start, end = mediator.normalize_timesteps(
        datetime_start,
        datetime_end,
        timestep = 5,
    )
    assert(start == datetime.datetime(2021, 1, 1, 1, 30, 0, tzinfo = tzinfo))
    assert(end == datetime.datetime(2021, 1, 3, 1, 30, 0, tzinfo = tzinfo))

    start, end = mediator.normalize_timesteps(
        datetime_start,
        datetime_end,
        timestep = 15,
    )
    assert(start == datetime.datetime(2021, 1, 1, 1, 30, 0, tzinfo = tzinfo))
    assert(end == datetime.datetime(2021, 1, 3, 1, 30, 0, tzinfo = tzinfo))
    return

# Tests for the forecasts.py module. They can be run with:
#  $ python manage.py test

from django.test import TestCase

import datetime
import pandas
from pvlib import location
import pytz

from mediation import forecasts
from mediation import models

class ForecastTestCase(TestCase):

    def test_forecaster_from_plant(self):
        plant = models.PlantConfig.objects.get(pk = 1)
        forecaster = forecasts.SolarForecast(
            plant.latitude,
            plant.longitude,
            plant.timezone_string,
            plant.elevation,
        )
        assert(type(forecaster.plant_location) == location.Location)
        assert(
            type(forecaster.forecast_uncertainty) == forecasts.ForecastUncertainty
        )
        return
    
    def test_latestForecast(self):
        forecaster = forecasts.SolarForecast(
            38.2,
            -117.4,
            'US/Pacific',
            100.0,
        )
        data = forecaster.latestForecast()
        assert(len(data) == 48)
        assert('clear_sky' in data.keys())
        assert('0.5' in data.keys())
        # Check clearsky bounds
        assert(min(data['clear_sky']) < 100)
        assert(max(data['clear_sky']) > 800)
        # Check maximum solar in reasonable units, not a ratio
        assert(max(data['1.0']) > 200)
        return

    def test_latestForecast_resolution(self):
        forecaster = forecasts.SolarForecast(
            38.2,
            -117.4,
            'US/Pacific',
            100.0,
        )
        data = forecaster.latestForecast(resolution = '2h')
        assert(len(data) == 24)
        return

    def test_latestForecast_horizon(self):
        forecaster = forecasts.SolarForecast(
            38.2,
            -117.4,
            'US/Pacific',
            100.0,
        )
        data = forecaster.latestForecast(horizon = pandas.Timedelta(hours = 24))
        assert(len(data) == 24)
        return

    def test_getForecast(self):
        forecaster = forecasts.SolarForecast(
            38.2,
            -117.4,
            'US/Pacific',
            100.0,
        )
        # Choose a start time that is not current, but that NDFD will still have
        # data for.
        datetime_start = datetime.datetime.now(pytz.timezone('US/Pacific'))
        data = forecaster.getForecast(
            datetime_start = datetime_start - pandas.Timedelta(hours = 24),
            horizon = pandas.Timedelta(hours = 24),
            resolution = pandas.Timedelta(minutes = 1),
        )
        assert(len(data) == 24 * 60)

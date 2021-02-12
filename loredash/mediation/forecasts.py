import datetime
import os
import pandas
from pvlib import forecast
from pvlib import location

from mediation.models import SolarForecastData

class ForecastUncertainty:
    """
    A class to manage the uncertainty bands of the DNI forecasts.

    Attributes
    ----------
    data : dataframe
    """
    data = None
    def __init__(self, filename):
        self.data = pandas.read_csv(filename)
        return
    
    def _interpolate(self, X, Y, median_estimate):
        last_x, last_y = 0.0, 0.0
        for (x, y) in zip(X, Y):
            if median_estimate <= x:
                w = (x - median_estimate) / (x - last_x)
                return w * last_y + (1 - w) * y
            else:
                last_x, last_y = x, y
        return Y[-1]
    
    def forecast(self, hours_ahead, median_estimate):
        """
        Parameters
        ----------
        hours_ahead : int
            Difference in hours between the current time and the time for when 
            the forecast applies.
        median_estimate : float
            The median estimate for the forecast.
        """
        T = self.data['horizon'][self.data['horizon'] <= hours_ahead].max()
        data = self.data[self.data['horizon'] == T]
        return data.groupby(by = ['quantile']).apply(
            lambda df: self._interpolate(
                df['from'].tolist(), df['to'].tolist(), median_estimate
            )
        )

class SolarForecast:
    """
    A class to manage DNI forecasts from NDFD.
    
    Attributes
    ----------
    plant_location : pvlib.location.Location
        The location of the plant.
    forecast_uncertainty : ForecastUncertainty
    """
    plant_location = None
    forecast_uncertainty = None
    def __init__(
        self,
        latitude,
        longitude,
        timezone,
        altitude,
        uncertainty_bands = ""
    ):
        """
        Parameters
        ----------
        latitude : float
            The latitude of the plant.
        longitude : float
            The longitude of the plant.
        timezone : float
            The timezone of the plant.
        altitude : float
            The altitude of the plant. Used to compute clear-sky DNI.
        uncertainty_bands : str
            The filename of a CSV file containing the uncertainty matrices.
        """
        self.plant_location = location.Location(
            latitude,
            longitude,
            timezone,
            altitude,
        )
        if uncertainty_bands == "":
            uncertainty_bands = os.path.join(
                os.path.dirname(__file__), '../data/solar_forecast_bands.csv'
            )
        self.forecast_uncertainty = ForecastUncertainty(uncertainty_bands)
        return
    
    def _rawData(self):
        """Return a Pandas dataframe of the current DNI forecast and 
        corresponding  clear-sky DNI estimate.
    
        Parameters
        ----------
        resolution : str, optional
            The temporal resolution to use when getting forecast.
        """
        current_time = pandas.Timestamp(
            datetime.datetime.now(),
            tz = self.plant_location.tz,
        )
        data = forecast.NDFD().get_processed_data(
            self.plant_location.latitude,
            self.plant_location.longitude,
            current_time,
            current_time + pandas.Timedelta(hours = 48),
            how = 'clearsky_scaling',
        )[['dni']]
        data.index = pandas.to_datetime(data.index)
        data['clear_sky'] = self.plant_location.get_clearsky(data.index)['dni']
        # Drop any rows at which NDFD predicts essentially zero DNI, or the
        # clear-sky is zero (e.g., night-time).
        data = data[(data['dni'] > 5) & (data['clear_sky'] > 5)]
        # Map the values to a normalized dni/clear-sky ratio space to allow us
        # to convert the median estimate from NDFD to a probabilistic estimate.
        data['ratio'] = data['dni'] / data['clear_sky']
        return data
    
    def _toUTC(self, t):
        "Convert a timezone-aware `t` to UTC and strip timezone info."
        return t.astimezone('UTC').replace(tzinfo = None)

    def _toLocal(self, t):
        "Convert a timezone-naive `t` to local timezone-aware."
        return t.tz_localize('UTC').astimezone(self.plant_location.tz)

    def updateDatabase(self):
        data = self._rawData()
        current_time = self._toUTC(
            pandas.Timestamp(
                datetime.datetime.now(), tz = self.plant_location.tz,
            )
        )
        instances = [
            SolarForecastData(
                # Note how these are now in UTC! Remember this when we get them
                # back.
                forecast_made = current_time,
                forecast_for = self._toUTC(time),
                clear_sky = row.clear_sky,
                ratio = row.ratio,
            )
            # Okay, I know data.iterrows is slow. But the dataframe is never 
            # very big.
            for (time, row) in data.iterrows()
        ]
        SolarForecastData.objects.bulk_create(instances, ignore_conflicts=True)
        return

    def _hourDiff(self, t):
        return (datetime.datetime.utcnow() - t).total_seconds() / 3600

    def _updateLatestForecast(self, resolution):
        self.updateDatabase()
        return self.latestForecast(resolution = resolution)

    def latestForecast(
        self, 
        resolution = '1h',
        update_threshold = 3,
    ):
        """
        Return the latest DNI forecast.

        Parameters
        ----------
        resolution : str
            The resolution passed to `pandas.ressample` for resampling the 
            NDFD forecast into finer resolution. Defaults to `1h`.
        update_threshold : int
            If the latest forecast was retrieved more than `update_threshold` 
            hours ago, refresh the database before returing the latest forecast.
        """
        # First, read latest forecast data from database, and convert it to the
        # plant's timezone.
        try:
            latest = SolarForecastData.objects.latest('forecast_made').forecast_made
        except SolarForecastData.DoesNotExist:
            return self._updateLatestForecast(resolution = resolution)
        if self._hourDiff(latest) >= update_threshold:
            return self._updateLatestForecast(resolution = resolution)
        raw_data = pandas.DataFrame(
            SolarForecastData.objects.filter(forecast_made = latest).values()
        )
        raw_data.drop(columns = ['forecast_made', 'id'], inplace=True)
        raw_data['forecast_for'] = \
            raw_data['forecast_for'].map(lambda x: self._toLocal(x))
        raw_data.set_index('forecast_for', inplace=True)
        data = raw_data.resample(resolution).mean()
        # Clean up the ratio by imputing any NaNs that arose to the nearest
        # non-NaN value. This means we assume that the start of the day acts
        # like the earliest observation, and the end of the day looks like the
        # last observation.
        data.interpolate(method = 'linear', inplace = True)
        # However, nearest only works when there are non-NaN values either side.
        # For the first and last NaNs, use bfill and ffill:
        data.fillna(method = 'bfill', inplace = True)
        data.fillna(method = 'ffill', inplace = True)
        # For each row (level = 0), convert the median ratio estimate into a 
        # probabilistic ratio estimate.
        data = data.groupby(level = 0).apply(
            lambda df: self.forecast_uncertainty.forecast(
                (df.index[0] - raw_data.index[0]).seconds / 3_600,
                df['ratio'][0],
            )
        )
        # Get clear-sky estimates for the new time-points.
        data['clear_sky'] = self.plant_location.get_clearsky(data.index)['dni']
        # Convert the ratio estimates back to DNI-space by multiplying by the 
        # clear-sky.
        for k in data.keys():
            if k == 'clear_sky':
                continue
            data[k] = data[k] * data['clear_sky']
        data.rename(columns = {k: str(k) for k in data.keys()}, inplace=True)
        return data

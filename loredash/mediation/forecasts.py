import datetime
import numpy
import os
import pandas
import pytz

# pvlib.forecasts issues a warning on init. Silence it.
import warnings
warnings.filterwarnings(
    'ignore',
    message = "The forecast module algorithms and features are highly experimental.",
)

from pvlib import forecast
from pvlib import location

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
        timezone : int
            The timezone of the plant in fixed-offset.
        altitude : float
            The altitude of the plant. Used to compute clear-sky DNI.
        uncertainty_bands : str
            The filename of a CSV file containing the uncertainty matrices.
        """
        if not isinstance(timezone, int):
            raise TypeError("Expected the timezone to be an integer representing the FixedOffset time.")
        self.plant_location = location.Location(
            latitude,
            longitude,
            pytz.FixedOffset(60 * timezone),
            altitude,
        )
        if uncertainty_bands == "":
            uncertainty_bands = os.path.join(
                os.path.dirname(__file__), '../data/solar_forecast_bands.csv'
            )
        self.forecast_uncertainty = ForecastUncertainty(uncertainty_bands)
        return
        
    def get_raw_data(self, datetime_start):
        """
        Get the forecast data from pvlib, beginning at `datetime_start`. This
        includes the probabilistic forecasts of DNI.
        """
        # This datetime_start coming in is in UTC.
        assert(datetime_start.tzinfo == pytz.UTC)
        include_columns = ['dni', 'ghi', 'dhi', 'temp_air', 'wind_speed']
        data = forecast.NDFD().get_processed_data(
            self.plant_location.latitude,
            self.plant_location.longitude,
            # Sufficiently long time window for NDFD. We also pick a time window
            # prior to the actual start time to work-around pvlib forecast
            # vagaries with how it returns forecasts.
            datetime_start - pandas.Timedelta(days = 1),
            datetime_start + pandas.Timedelta(days = 7),
            how = 'clearsky_scaling',
        )[include_columns]
        data.index = pandas.to_datetime(data.index)
        # Strip out time periods prior to the datetime_start.
        first_index = 0
        for i in range(len(data)):
            if data.index[1] >= datetime_start:
                first_index = i - 1
        data = data[first_index:]
        data['clear_sky'] = self.plant_location.get_clearsky(data.index)['dni']
        # Map the values to a normalized dni/clear-sky ratio space to allow us
        # to convert the median estimate from NDFD to a probabilistic estimate.
        data['ratio'] = data['dni'] / data['clear_sky']
        # To avoid weirdness, ignore rows in early morning and early evening.
        # We will impute their values from neighbours.
        data.loc[data['clear_sky'] < 200, 'ratio'] = numpy.nan
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
        probabilistic_data = data.groupby(level = 0).apply(
            lambda df: self.forecast_uncertainty.forecast(
                (df.index[0] - data.index[0]).seconds / 3_600,
                df['ratio'][0],
            )
        )
        new_data = probabilistic_data.join(data)
        forecast_columns = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        new_data.rename(
            columns = {k: str(k) for k in forecast_columns}, 
            inplace=True,
        )
        for k in forecast_columns:
            new_data[str(k)] = new_data[str(k)] * new_data['clear_sky']
        return new_data

import datetime
import pandas
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

class Forecast:
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
        latitude = 38.238875,
        longitude = -117.363598,
        timezone = 'US/Pacific',
        altitude = 1497.0,
        uncertainty_bands = 'bands.csv',
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
        self.forecast_uncertainty = ForecastUncertainty(uncertainty_bands)
        return
    
    def _raw_data(self):
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
        # Map the values to a normalized dni/clear-sky ratio space to allow us
        # to convert the median estimate from NDFD to a probabilistic estimate.
        data['clear_sky'] = self.plant_location.get_clearsky(data.index)['dni']
        data['ratio'] = data['dni'] / data['clear_sky']
        # Clean up the ratio by imputing any NaNs that arose to the nearest 
        # non-NaN value. This means we assume that the start of the day acts 
        # like the earliest observation, and the end of the day looks like the 
        # last observation.
        data.interpolate(method = 'nearest', inplace = True)
        # However, nearest only works when there are non-NaN values either side.
        # For the first and last NaNs, use bfill and ffill:
        data.fillna(method = 'bfill', inplace = True)
        data.fillna(method = 'ffill', inplace = True)
        return data
    
    def forecast(self, resolution = '5T'):
        # Get the median estimate from NDFD, as well as the estimate in 
        # ratio-space.
        raw_data = self._raw_data()
        # For each row (level = 0), convert the median ratio estimate into a 
        # probabilistic ratio estimate.
        quantile_data = raw_data.groupby(level = 0).apply(
            lambda df: self.forecast_uncertainty.forecast(
                (df.index[0] - raw_data.index[0]).seconds / 3_600,
                df['ratio'][0],
            )
        )
        # Then, resample these ratio estimates to the user-provided resolution.
        data = quantile_data.resample(resolution).mean()
        data.interpolate(method = 'nearest', inplace = True)
        # Get clear-sky estimates for the new time-points.
        data['clear_sky'] = self.plant_location.get_clearsky(data.index)['dni']
        # Convert the ratio estimates back to DNI-space by multiplying by the 
        # clear-sky.
        for k in data.keys():
            if k == 'clear_sky':
                continue
            data[k] = data[k] * data['clear_sky']
        return data

# f = Forecast()
# d = f.forecast()

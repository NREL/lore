import numpy
import pandas
from pvlib import forecast
from pvlib import location

def _ndfd_to_pandas_date(ndfd_date, time_zone = 'UTC'):
    y = []
    for shift in (100, 100, 100, 100, 10000):
        y.append(int(ndfd_date % shift))
        ndfd_date -= y[-1]
        ndfd_date /= shift
    assert ndfd_date == 0.0
    return pandas.Timestamp(
        year = y[4], 
        month = y[3], 
        day = y[2], 
        hour = y[1],
        minute = y[0],
        tz = time_zone
    )

def _read_ndfd_data(model, sky_cover_filename):
    data = pandas.read_csv(
        sky_cover_filename,
        header = None, 
        names = ['FORECAST_MADE', 'FORECAST_FOR', 'SKY_COVER']
    )
    # There are some missing values, indicated by sky cover of 9999.
    data.loc[data['SKY_COVER'] > 101, 'SKY_COVER'] = None
    # Then kill any NaN's 
    data.fillna(method = 'ffill', inplace = True)
    # Convert the NDFD times to a pandas `Datetime`.
    for key in ['FORECAST_MADE', 'FORECAST_FOR']:
        data[key] = pandas.DatetimeIndex(
            data[key].apply(_ndfd_to_pandas_date)
        ).tz_convert(model.location.tz)
    # To tidy the dataframe, sort it.
    data.sort_values(['FORECAST_MADE', 'FORECAST_FOR'], inplace = True)
    # Okay. So NDFD make forecasts in 3 hr increments. But we really want
    # hourly forecasts, so we're going to re-sample into 1 hour intervals 
    # and and then linearly interpolate between forecasts.
    #
    # Along the way, we need to account for daylight saving. These hours are
    # essentially duplicates (e.g., two entries at 1 am for 4 am). Luckily, 
    # these occur at night when the sun is not shining! Handle these by 
    # averaging any duplicates.
    data = data.groupby(
        ['FORECAST_MADE', 'FORECAST_FOR']
    ).mean(
    ).reset_index(        
    ).groupby(
        ['FORECAST_MADE']
    ).apply(
        lambda df: df.set_index(
            'FORECAST_FOR'
        ).resample(
            '1h'
        ).mean(
        ).interpolate(
        )
    ).reset_index(
    )
    # Convert the sky cover to a DNI value, making sure that we use the 
    # `FORECAST_FOR` time, and that time-zones etc. play nice.
    data.set_index(['FORECAST_FOR'], inplace = True)
    data['PREDICTED_DNI'] = model.cloud_cover_to_irradiance(
        data['SKY_COVER'], 
        how = 'clearsky_scaling',
    )['dni']
    data.reset_index(inplace = True)
    return data

def _read_historical_data(model, historical_filename):
    data = pandas.read_csv(historical_filename)
    data['FORECAST_FOR'] = pandas.DatetimeIndex(
        data['FORECAST_FOR'].apply(
            lambda d: _ndfd_to_pandas_date(d, model.location.tz)
        )
    )
    return data

def _joined_dataframe(model, ndfd_filename, historical_filename):
    df = _read_ndfd_data(
        model, ndfd_filename
    ).set_index(
        'FORECAST_FOR'
    ).join(
        _read_historical_data(
            model, historical_filename
        ).set_index(
            'FORECAST_FOR'
        )
    )
    df['HOUR_DIFF'] = (df.index - df['FORECAST_MADE']).apply(
        lambda x: int(x.total_seconds() / 3600)
    )
    df['MEASURED_RATIO'] = df['MEASURED_DNI'] / df['CLEAR_SKY']
    df['PREDICTED_RATIO'] = df['PREDICTED_DNI'] / df['CLEAR_SKY']
    return df

def _compute_transition_matrix_inner(df, quantiles):
    def _compute_quantile(df, lo, hi, quantiles):
        row_filter = \
            (lo <= df['PREDICTED_RATIO']) & \
            (df['PREDICTED_RATIO'] <= hi)
        to = df[row_filter]['MEASURED_RATIO'].quantile(quantiles)
        return pandas.DataFrame({
            'quantile': quantiles,
            'from': hi,
            'to': list(to),
        })
    d = pandas.cut(df['PREDICTED_RATIO'], bins=10, retbins=True)[1]
    df = pandas.concat([
        _compute_quantile(df, d[i - 1], d[i], quantiles) 
        for i in range(1, len(d))
    ]).reset_index()
    df.drop(columns = 'index', inplace = True)
    df.sort_values(['quantile', 'from'], inplace = True)
    return df

def _compute_transition_matrix(df, aggregation_window, quantiles):
    df['HOUR_DIFF_AGG'] = df['HOUR_DIFF'].apply(
        lambda x: aggregation_window * round(x / aggregation_window)
    )
    t_mat = df.groupby(['HOUR_DIFF_AGG']).apply(
        lambda d: _compute_transition_matrix_inner(d, quantiles)
    ).reset_index()
    t_mat.drop(columns = 'level_1', inplace = True)
    t_mat.rename(columns = {'HOUR_DIFF_AGG': 'horizon'}, inplace = True)
    return t_mat

def compute_transition_matrix(
    ndfd_filename,
    historical_filename,
    location,
    output_filename = 'bands.csv',
    minimum_clear_sky = 50,
    forecast_horizon = 60,
    aggregation_window = 12,
    quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
):
    model = forecast.NDFD()
    model.location = location
    df = _joined_dataframe(model, ndfd_filename, historical_filename)
    # Drop forecasts when clear-sky is too low. It's easy to forecast 0 DNI at 
    # night! And only have forecasts go out to a sensible horizon.
    df = df[
        (df['CLEAR_SKY'] >= minimum_clear_sky) & \
        (df['HOUR_DIFF'] <= forecast_horizon)
    ]
    t_mat = _compute_transition_matrix(df, aggregation_window, quantiles)
    t_mat.to_csv(output_filename, index=False)
    return t_mat


### ============================================================================
### Main code goes here.
### ============================================================================

t_mat = compute_transition_matrix(
    ndfd_filename = 'sky_cover.csv',
    historical_filename = 'historical_dni.csv',
    location = location.Location(
        38.23889,
        -117.36358,
        'US/Pacific',
        1500.0,
    ),
)

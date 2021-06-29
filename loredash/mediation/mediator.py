from django.conf import settings
from django.db import IntegrityError
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import time, copy, datetime, math, pytz
from twisted.internet.task import LoopingCall
from twisted.internet import reactor
import pandas as pd
from pandas.tseries.frequencies import to_offset
import numpy as np
import rapidjson

from data.mspt_2020_defaults import default_ssc_params
from mediation import tech_wrap, data_validator, dispatch_wrap, models, forecasts
import mediation.plant as plant_
from mediation.plant import Revenue

class Mediator:
    tech_wrap = None
    validated_outputs_prev = None

    def __init__(self, params, plant_design_path, weather_file=None,
                 update_interval=datetime.timedelta(seconds=5)):
        self.params = params
        self.weather_file = weather_file
        self.update_interval = update_interval
        self.simulation_timestep = datetime.timedelta(hours=1/self.params['time_steps_per_hour'])
        self.params['dispatch_factors_ts'] = Revenue.get_price_data(
            self.params['price_multiplier_file'],
            self.params['avg_price'],
            self.params['price_steps_per_hour'],
            self.params['time_steps_per_hour'])

        with open(plant_design_path) as f:
            plant_design = rapidjson.load(f)
        self.plant = plant_.Plant(
            design=plant_design,
            initial_state=plant_.plant_initial_state)

        default_ssc_params.update(self.plant.get_state())                       # combine default and plant params, overwriting the defaults
        default_ssc_params.update(self.params)                                  # combine default and mediator params, overwriting the defaults
        self.tech_wrap = tech_wrap.TechWrap(
            params=default_ssc_params,                                          # already a copy so tech_wrap cannot edit
            plant=copy.deepcopy(self.plant),                                    # copy so tech_wrap cannot edit
            dispatch_wrap_params=dispatch_wrap.dispatch_wrap_params.copy(),     # copy so tech_wrap cannot edit
            weather_file=None)

        self.forecaster = forecasts.SolarForecast(
            self.plant.design['latitude'],
            self.plant.design['longitude'],
            self.plant.design['timezone_string'],
            self.plant.design['elevation'])

        self.plant.update_flux_maps(self.tech_wrap.calc_flux_eta_maps(self.plant.get_design(), self.plant.get_state()))

        # Setup dispatch_wrap
        dispatch_wrap_params = dispatch_wrap.dispatch_wrap_params                                   # TODO: replace with a path to a JSON config file
        dispatch_wrap_params.update(self.params)                                                    # include mediator params in with dispatch_wrap_params
        self.dispatch_wrap = dispatch_wrap.DispatchWrap(plant=self.plant, params=dispatch_wrap.dispatch_wrap_params)
    
    def run_once(self, datetime_start, datetime_end):
        """Get data from external plant and weather interfaces and run entire set
        of submodels, saving data to database.

        Args:
            datetime_start: beginning of first timestep, in plant-local time
            datetime_end:   end of last timestep, in plant-local time
        Returns:
            0 if successful
        Raises:
        """
        # Code Design:
        # Step 0: Normalize timesteps to even intervals and enforce timestep localization
        #
        # Step 1:
        #   Thread 1:
        #       a. Call virtual/real plant to get plant operating state and any local weather data
        #       b. Validate these data
        #       c. Store in database and add to current timestep cache
        #   Thread 2:
        #       a. Call module to retrieve weather data(s)
        #       b. Validate these data
        #       c. Store in database and add to current timestep cache
        #
        # Step 2:
        #   Thread 1:
        #       a. Call dispatch model
        #       b. Validate these data
        #       c. Store in database and add to current timestep cache
        #
        # Step 3:
        #   Thread 1:
        #       a. Set dispatch targets in tech model
        #       b. Call tech model using inputs
        #       c. Validate output data
        #       d. Add simulated plant state and other data to cache and store in database
        #
        # Step 4:
        #   Thread 1:
        #       a. Update previous timestep cache with current timestep cache and then clear current cache


        # Step 0: Normalize timesteps to even intervals and enforce timestep localization
        if datetime_start.tzinfo is None or datetime_end.tzinfo is None or \
            datetime_start.tzinfo != datetime_end.tzinfo:
            print("WARNING: Timesteps in run_once were not properly localized.")
            _tz = pytz.timezone(self.plant.design['timezone_string'])
            datetime_start = _tz.localize(datetime_start)
            datetime_end = _tz.localize(datetime_end)
        self._validate_plant_local_time(datetime_start)
        self._validate_plant_local_time(datetime_end)

        datetime_start, datetime_end = normalize_timesteps(
            datetime_start,
            datetime_end,
            timestep = self.simulation_timestep.seconds / 60)

        # Step 1, Thread 1:
        # a. Call virtual/real plant to get plant operating state and any local
        #    weather data
        plant_state = self.plant.get_state()

        # b. Validate these data
            #TODO: Add this

        # c. Store in database and add to current timestep cache
            #TODO: Add this

        # Step 1, Thread 2:
        # a. Get weather data and forecasts
        datetime_end_dispatch = datetime_start + \
            datetime.timedelta(hours=self.dispatch_wrap.params['dispatch_horizon'])
        weather_dispatch = self.get_weather_df(
            datetime_start=datetime_start,
            datetime_end=datetime_end_dispatch,
            timestep=datetime.timedelta(minutes=min(self.dispatch_wrap.params['dispatch_steplength_array'])),
            tmy3_path=self.weather_file,
            use_forecast=True)
        # Sanity check to make sure this is what we expect.
        assert(weather_dispatch.index[0] == datetime_start)
        assert(weather_dispatch.index[-1] == datetime_end_dispatch)
        weather_simulate = self.get_weather_df(
            datetime_start=datetime_start,
            datetime_end=datetime_end,
            timestep=datetime.timedelta(hours=1/self.params['time_steps_per_hour']),
            tmy3_path=self.weather_file,
            use_forecast=False)          # TODO: this use_forecast should probably be True too. Can we optimize these two get_weather_df calls
                                            #       since getting the forecasts takes significant time?
        # Sanity check to make sure this is what we expect.
        assert(weather_simulate.index[0] == datetime_start)
        assert(weather_simulate.index[-1] == datetime_end)
        # Set clearsky data
        clearsky_data = np.nan_to_num(np.array(weather_dispatch['Clear Sky DNI']), nan = 0.0)  # TODO: Better handling of nan values in clear-sky data
        clearsky_data_padded = self.tech_wrap.pad_weather_data(list_data = clearsky_data.tolist(), 
            datetime_start = self._toTMYTime(datetime_start), 
            timestep = datetime.timedelta(hours=1/self.params['time_steps_per_hour']))
        self.tech_wrap.set({'rec_clearsky_dni': clearsky_data_padded})

        # TODO(odow): keep pushing timezones through the code.
        datetime_start = datetime_start.replace(tzinfo = None)
        datetime_end = datetime_end.replace(tzinfo = None)

        # Step 2, Thread 1:
        # a. Call dispatch model, (which includes the 'f_estimates...' tech_wrap function to get estimates) and update inputs for next call
        # TODO: Calls to ssc need to be fixed-offset time, but need to be careful with tech_wrap.set_weather_data()
        dispatch_outputs = self.dispatch_wrap.run(
            datetime_start=datetime_start,
            ssc_horizon = (datetime_end - datetime_start),
            weather_dataframe=weather_dispatch,
            f_estimates_for_dispatch_model=self.tech_wrap.estimates_for_dispatch_model,
            update_interval = 1.0/self.params['time_steps_per_hour'],
            initial_plant_state=plant_state
        )

        # b. Validate these data
        dispatch_outputs = data_validator.validate(dispatch_outputs, data_validator.dispatch_outputs_schema)
        ssc_dispatch_targets = data_validator.validate(dispatch_outputs['ssc_dispatch_targets'].asdict(use_lists=True),
                                                       data_validator.ssc_dispatch_targets_schema)
        dispatch_outputs['ssc_dispatch_targets'].update_from_dict(ssc_dispatch_targets)

        # c. Store in database and add to current timestep cache
            #TODO: Add this


        # Step 3, Thread 1:
        # a. Set dispatch targets in tech model
        self.tech_wrap.set(dispatch_outputs['ssc_dispatch_targets'].asdict())

        # b. Call tech model using inputs
        tech_outputs = self.tech_wrap.simulate(
            datetime_start,
            datetime_end,
            self.simulation_timestep,
            plant_state,
            weather_dataframe=weather_simulate)
        print("Generated Energy [kWh]= ", tech_outputs["annual_energy"])

        # c. Validate output data
        tech_outputs = {k:(list(v) if isinstance(v, tuple) else v) for (k,v) in tech_outputs.items()}   # converts tuples to lists so they can be edited
        tic = time.process_time()
        validated_outputs = data_validator.validate(tech_outputs, data_validator.ssc_schema)
        toc = time.process_time()
        print("Validation took {seconds:0.2f} seconds".format(seconds=toc-tic))
        validated_outputs['year_start'] = datetime_start.year                                           # add this so date can be determined from time_hr

        # d. Add simulated plant state and other data to cache and store in database
        self.validated_outputs_prev = copy.deepcopy(validated_outputs)
        new_plant_state_vars = self.tech_wrap.get_simulated_plant_state(validated_outputs)             # for initializing next simulation from a prior one
        self.plant.update_state(validated_outputs, new_plant_state_vars, self.simulation_timestep.seconds/3600)
        self.bulk_add_to_db_table(validated_outputs)                                                 # add to database

        return 0

    def run_continuously(self, update_interval=5):
        """Continuously get data from external plant and weather interfaces and run
        entire set of submodels, saving data to database
        
        update_interval -- [s] how frequently the interfaces and submodels are polled and run, respectively
        """
        looping_call = LoopingCall(self.run_once)
        time.sleep(update_interval - time.time() % update_interval)          # wait to start until it's an even clock interval
        looping_call.start(update_interval)
        reactor.run()

    def get_current_plant_time(self):
        "Return the current time in plant-local time."
        return datetime.datetime.now(
            pytz.timezone(self.plant.design['timezone_string']),
        )

    def model_previous_day_and_add_to_db(self):
        """Simulate previous day and add to database
        e.g.:
        if current time is 17:43 and simulation_timestep = 5 minutes:
            it will model from 17:40 yesterday (start of timestep)
            to 17:40 today (end of timestep)
            with timesteps in database (end of timesteps) being from 17:45 yesterday to 17:40 today
            for 288 total new entries
        """
        # Make sure the time is localized to the timezone of the plant!
        datetime_now = self.get_current_plant_time()
        datetime_now_rounded_down = round_minutes(datetime_now, 'down', self.simulation_timestep.seconds/60)    # the start of the time interval currently in
        datetime_start_prev_day = datetime_now_rounded_down - datetime.timedelta(days=1)
        datetime_end_current_day = datetime_now_rounded_down                   # end of the last timestep
                                                                            # (as noted for "time_stop" on line 1004 in cmod_tcsmolten_salt.cpp)
        self.run_once(datetime_start_prev_day, datetime_end_current_day)
        return 0
    
    def _validate_plant_local_time(self, time):
        """A helper function that validates the given `time` is localized to the
        timezone of the plant.
        """
        assert(time.tzinfo.zone == self.plant.design['timezone_string'])
        return

    def bulk_add_to_db_table(self, records):
        n_records = len(records['time_hr'])
        newyears = datetime.datetime(records['year_start'], 1, 1, 0, 0, 0)

        instances = [
            models.TechData(
                timestamp =             round_time(newyears + datetime.timedelta(hours=records['time_hr'][i]), 1),       # round to nearest second
                E_tes_charged =         records['e_ch_tes'][i],
                eta_tower_thermal =     records['eta_therm'][i],
                eta_field_optical =     records['eta_field'][i],
                W_grid_no_derate =      records['P_out_net'][i],
                tou =                   records['tou_value'][i],
                W_grid_with_derate =    records['gen'][i],
                Q_tower_incident =      records['q_dot_rec_inc'][i],
                Q_field_incident =      records['q_sf_inc'][i],
                pricing_multiple =      records['pricing_mult'][i],
                dni =                   records['beam'][i],
            )
            for i in range(n_records)
        ]

        try:
            models.TechData.objects.bulk_create(instances, ignore_conflicts=True)
            # If ignore_conflicts=False and if any to-be-added records are already in the database, as indicated by the timestamp,
            #  an exception is raised and no to-be-added records are added.
            # If ignore_conflicts=True, all records not already in the database are added. To-be-added records that are already in the
            #  database do not replace the database records. Therefore, no existing database records are overwritten.
        except IntegrityError as err:
            error_string = format(err)
            if error_string == "UNIQUE constraint failed: mediation_techdata.timestamp":
                raise IntegrityError(error_string)      # just re-raise the exception for now
        except Exception as err:
            raise(err)
    
    def _toTMYTime(self, time):
        "Convert a time to TMY timezone, which uses a fixed offset."
        fixed_tz = pytz.FixedOffset(60 * self.plant.design['timezone'])
        return time.astimezone(fixed_tz).replace(tzinfo=None)

    def get_weather_df(
        self,
        datetime_start,
        datetime_end,
        timestep,
        tmy3_path,
        use_forecast=False,
    ):
        """
        Return a dataframe of weather data at `timestep` resolution (of
        at-most 1 hour) covering the time-span given by `datetime_start` and
        `datetime_end`.

        Datetimes are in plant-local time.

        If `use_forecast`, replace the 'DNI', 'DHI', 'GHI', 'Wind Speed' columns
        with the latest NDFD forecast from the forecasts submodule. In addition,
        create two new columns: 'Clear Sky DNI' and 'Ambient Temperature'.
        """
        self._validate_plant_local_time(datetime_start)
        self._validate_plant_local_time(datetime_end)
        assert(timestep.total_seconds() <= 3600)
        # The timezone of the TMY file:
        tmy_tz = pytz.FixedOffset(60 * self.plant.design['timezone'])
        # Get the TMY data. Make sure to convert the timezones into the TMY
        # timezone (fixed offset), and then to strip the timezone data!
        data = tmy3_to_df(
            tmy3_path,
            datetime_start.astimezone(tmy_tz).replace(tzinfo=None),
            datetime_end.astimezone(tmy_tz).replace(tzinfo=None),
        )
        # Re-localize the datetimes again. First add the TMY timezone, and then
        # convert to plant-local time.
        data.index = data.index.tz_localize(tmy_tz).tz_convert(datetime_start.tzinfo)
        # Resample data into finer timesteps, filling with previous value.
        data = data.resample(timestep).pad()
        # Extrapolate out last point, filling with last value
        dates = data.index
        n_periods = pd.to_timedelta(to_offset(pd.infer_freq(data.index))) / dates.freq - 1
        dates = dates.union(
            pd.date_range(
                start=dates[-1] + dates.freq,
                periods=n_periods,
                freq=dates.freq,
            ),
        )
        data = data.reindex(dates)
        data.fillna(method = 'ffill', inplace = True)
        # Now strip the data back to what the user asked for:
        data = data[(data.index >= datetime_start) & (data.index <= datetime_end)]
        if use_forecast:
            tic = time.process_time()
            solar_forecast = self.forecaster.getForecast(
                datetime_start=data.index[0],
                horizon=data.index[-1] - data.index[0],
                resolution=timestep)
            toc = time.process_time()
            print("Generating forecast took {seconds:.2f} seconds".format(seconds=toc-tic))
            key_map = {
                'dni': 'DNI',
                'dhi': 'DHI',
                'ghi': 'GHI',
                'wind_speed': 'Wind Speed',
                # The TMY data has a column named 'Temperature'.
                # This is different.
                'temp_air': 'Ambient Temperature',
                # This is not part of the TMY file!
                'clear_sky': 'Clear Sky DNI',
            }
            for (k, v) in key_map.items():
                data.loc[:, v] = list(solar_forecast[k])
        return data

def mediate_continuously(update_interval=5):
    mediator = Mediator()
    mediator.run_continuously(update_interval=update_interval)
    return False

# def mediate_once():
#     """This will likely only be used for testing"""
#     mediator = Mediator()
#     mediator.RunOnce()
#     return False

def round_time(dt, second_resolution):
    """Round to nearest second interval"""
    seconds = (dt.replace(tzinfo=None) - dt.min).seconds + dt.microsecond * 1.e-6
    rounding = (seconds+second_resolution/2) // second_resolution * second_resolution
    return dt + datetime.timedelta(0,rounding-seconds,0)

def round_minutes(dt, direction, minute_resolution):
    """Round to nearest minute interval
    e.g.:
        dt = datetime.datetime(2021, 1, 4, 15, 22, 0, 9155)
        direction = up
        minute_resolution = 5
        -----
        result = datetime.datetime(2021, 1, 4, 15, 25, 0, 0)
    """
    on_interval = math.isclose((dt.minute + dt.second/60) % minute_resolution, 0., rel_tol=1e-6)
    new_minute = (dt.minute // minute_resolution + (1 if direction == 'up' and not on_interval else 0)) * minute_resolution
    new_time_old_seconds = dt + datetime.timedelta(minutes=new_minute - dt.minute)
    return new_time_old_seconds.replace(second=0, microsecond=0)

def normalize_timesteps(datetime_start, datetime_end, timestep):
    """Normalize the start and end datetimes to an integer multiple of the
    timestep [minutes].
    """
    new_start = round_minutes(datetime_start, 'down', timestep)
    new_end = round_minutes(datetime_end, 'up', timestep)
    print("Requested start:  {datetime}".format(datetime = datetime_start))
    print("Normalized start: {datetime}".format(datetime = new_start))
    print("Normalized end:   {datetime}".format(datetime = new_end))
    return new_start, new_end

def tmy3_to_df(tmy3_path, datetime_start=None, datetime_end=None):
    """does not work for end dates more than one year after start date"""

    if not isinstance(tmy3_path, str) or not os.path.isfile(tmy3_path):
        raise Exception('Tmy3 file not found')

    default_datetime_start = datetime.datetime(2018, 1, 1, 0, 0, 0)
    default_datetime_end = datetime.datetime(2018, 12, 31, 23, 59, 59)

    if datetime_start is None:
        datetime_start = default_datetime_start
    elif not isinstance(datetime_start, datetime.datetime):
        datetime_start = default_datetime_start
        print("The start datetime is not valid, using {date}.".format(date=default_datetime_start.strftime("%x")))

    if datetime_end is None:
        datetime_end = default_datetime_end
    elif not isinstance(datetime_end, datetime.datetime):
        datetime_end = default_datetime_end
        print("The end datetime is not valid, using {date}.".format(date=default_datetime_end.strftime("%x")))

    df = pd.read_csv(tmy3_path, sep=',', skiprows=2, header=0, index_col='datetime', \
        parse_dates={'datetime': [0, 1, 2, 3, 4]}, \
        date_parser=lambda x: datetime.datetime.strptime(x, '%Y %m %d %H %M'))
    df.index = df.index.map(lambda t: t.replace(year=df.index[0].year))     # normalize all years to that of 1/1. Could also do this via date_parser.
    df = df[df.columns.drop(list(df.filter(regex='Unnamed')))]      # drop unnamed columns (which are empty)
    location = get_weatherfile_location(tmy3_path)
    df.attrs.update(location)

    # Ensure the requested datetimes are contained within the returned weather datetimes
    df_timestep = df.index[1] - df.index[0]
    datetime_start_adj = round_minutes(datetime_start, 'down', df_timestep.seconds/60)
    datetime_end_adj = round_minutes(datetime_end, 'up', df_timestep.seconds/60)
    timestamp_start_query = pd.Timestamp(datetime_start_adj.replace(year=df.index[0].year))        # align the years and convert to Timestamp
    timestamp_end_query = pd.Timestamp(datetime_end_adj.replace(year=df.index[0].year))

    if timestamp_end_query >= timestamp_start_query:     # if dates aren't in different years
        df_out = df[timestamp_start_query:timestamp_end_query]
        df_out.index = df_out.index.map(lambda t: t.replace(year=datetime_start_adj.year))
    else:
        newyearsday = datetime.datetime(df.index[0].year, 1, 1, 0, 0, 0)
        newyearseve = (newyearsday - df_timestep).replace(year=df.index[0].year)
        df_out_endofyear = df[timestamp_start_query:newyearseve]
        df_out_endofyear.index = df_out_endofyear.index.map(lambda t: t.replace(year=datetime_start_adj.year))
        df_out_startofyear = df[newyearsday:timestamp_end_query]
        df_out_startofyear.index = df_out_startofyear.index.map(lambda t: t.replace(year=datetime_end_adj.year))
        df_out = pd.concat([df_out_endofyear, df_out_startofyear])

    df_out.attrs = df.attrs
    return df_out

def get_weatherfile_location(tmy3_path):
    df_meta = pd.read_csv(tmy3_path, sep=',', header=0, nrows=1)
    return {
        'latitude': float(df_meta['Latitude'][0]),
        'longitude': float(df_meta['Longitude'][0]),
        'timezone': int(df_meta['Time Zone'][0]),
        'elevation': float(df_meta['Elevation'][0])
    }

# This is duplicated in case_study.py
mediator_params = {
    'start_date_year':                      2018,                   # TODO: Remove the need for this somewhat arbitrary year

    # Control conditions
	'time_steps_per_hour':			        12,			            # Simulation time resolution in ssc (1min)   DUPLICATED to: ssc_time_steps_per_hour
	'is_dispatch':					        0,                      # Always disable dispatch optimization in ssc
	'is_dispatch_targets':			        True,		            # True if (is_optimize or control_cycle == 'CD_data')
    'is_optimize':					        True,                   # Use dispatch optimization
    'q_pc_target_su_in':                    [0],
    'q_pc_target_on_in':                    [0],
    'q_pc_max_in':                          [0],
    'is_rec_su_allowed_in':                 [0],
    'is_rec_sb_allowed_in':                 [0],
    'is_pc_su_allowed_in':                  [0],
    'is_pc_sb_allowed_in':                  [0],
	'control_field':				        'ssc',                  #'CD_data' = use CD data to control heliostats tracking, heliostats offline, and heliostat daily reflectivity.  Receiver startup time is set to zero so that simulated receiver starts when CD tracking begins
                                                                    #'ssc' = allow ssc to control heliostat field operations, assuming all heliostats are available

	'control_receiver':				        'ssc_clearsky',         #'CD_data' = use CD data to control receiver mass flow. Receiver startup time is set to zero so that simulated receiver starts when CD receiver finishes startup
                                                                    #'ssc_clearsky' = use expected clearsky DNI to control receiver mass flow.  If field_control = 'ssc' then the historical median startup time from CD data will be used to control receiver startup
                                                                    #'ssc_actual_dni' = use actual DNI to control receiver mass flow.  If field_control = 'ssc' then the historical median startup time from CD data will be used to control receiver startup   

	'control_cycle':				        'ssc_heuristic',        # Only used if is_optimize = False
                                                                    # 'CD_data' = use CD actual cycle operation to control cycle dispatch targets
                                                                    # 'ssc_heuristic' = allow ssc heuristic (no consideration of TOD price) to control cycle dispatch

    # Price
    'price_multiplier_file':                '../data/prices_flat.csv',
	'ppa_multiplier_model':			        1,
	'price_steps_per_hour':			        1,                      # Number of steps per hour in electricity price multipliers
	'avg_price':					        138,                    # Average electricity price ($/MWh):  CD original PPA was $138/MWh
    'avg_purchase_price':                   30,                     # Average electricity purchase price ($/MWh) (note, this isn't currently used in the dispatch model)
    'avg_price_disp_storage_incentive':     0.0,                    # Average electricity price ($/MWh) used in dispatch model storage inventory incentive

    # Field, receiver, and cycle simulation options
    # 'clearsky_file':                        './model-validation/input_files/weather_files/clearsky_pvlib_ineichen_1min_2018.csv',   # Expected clear-sky DNI from Ineichen model (via pvlib). 
    'clearsky_file':                        '../data/clearsky_pvlib_ineichen_1min_2018.csv',             # Expected clear-sky DNI from Ineichen model (via pvlib). 
    'CD_mflow_path1_file':                  './model-validation/input_files/mflow_path1_2018_1min.csv',  # File containing CD data for receiver path 1 mass flow rate (note, all values are zeros on days without data)
    'CD_mflow_path2_file':                  './model-validation/input_files/mflow_path2_2018_1min.csv',  # File containing CD data for receiver path 2 mass flow rate (note, all values are zeros on days without data)
    'CD_raw_data_direc':                    './input_files/CD_raw',                                      # Directory containing raw data files from CD
    'CD_processed_data_direc':              './input_files/CD_processed',                                # Directory containing files with 1min data already extracted
	'rec_control_per_path':			        True,
	'field_model_type':				        3,
	'eta_map_aod_format':			        False,
	'is_rec_user_mflow':			        False,		            # or this should be unassigned, True if control_receiver == 'CD_data'
	'rec_clearsky_fraction':		        1.0,
    'rec_clearsky_model':                   0,
    'rec_su_delay':                         0.,                     # = 0.01 if control_field == 'CD_data' or control_receiver == 'CD_data'  Set receiver start time and energy to near zero to enforce CD receiver startup timing
    'rec_qf_delay':                         0.,                     # = 0.01 if control_field == 'CD_data' or control_receiver == 'CD_data'
    'is_rec_to_coldtank_allowed':           True,
    'use_CD_measured_reflectivity':	        False,                  # Use measured heliostat reflectivity from CD data
    'fixed_soiling_loss':			        0.02,                   # Fixed soiling loss (if not using CD measured reflectivity) = 1 - (reflectivity / clean_reflectivity)
	'is_rec_startup_trans':			        False,                  # TODO: Disabling transient startup -> ssc not yet configured to start/stop calculations in the middle of startup with this model
	'is_rec_model_trans':			        False,                  # TODO: Disabling transient receiver model -> ssc not yet configured to store/retrieve receiver temperature profiles
    'cycle_type':                           'user_defined',         # 'user-defined', 'sliding', or 'fixed'
}

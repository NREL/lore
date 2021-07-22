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
from pathlib import Path

# import django     # Needed here before "import models" if there's ever error "Apps aren't loaded yet"
# django.setup()
from mediation import tech_wrap, data_validator, dispatch_wrap, models, forecasts
import mediation.plant as plant_
from mediation.plant import Revenue
# import multiprocessing

def run_lore():
    try:
        init_and_mediate()
    except OSError as err:
        print("ERROR: OS error: {0}".format(err))
    except Exception as err:
        print("ERROR: {0}".format(err))

def init_and_mediate():
    print("Initializing models...")
    parent_dir = str(Path(__file__).parents[1])
    default_weather_file = parent_dir + "/data/daggett_ca_34.865371_-116.783023_psmv3_60_tmy.csv"
    plant_design_path = parent_dir + "/config/plant_design.json"
    mediator_params_path = parent_dir + "/config/mediator_settings.json"
    dispatch_params_path = parent_dir + "/config/dispatch_settings.json"
    m = Mediator(
        params_path=mediator_params_path,
        plant_design_path=plant_design_path,
        weather_file=default_weather_file,
        dispatch_params_path=dispatch_params_path,
        update_interval=datetime.timedelta(seconds=5),
    )
    print("Modeling previous day...")
    result = m.model_previous_day_and_add_to_db()
    print("Finished modeling previous day.")

    print("Modeling next periods...")
    if settings.RUNNING_DEVSERVER == True:      # just for development testing
        result = m.run_once(
            datetime_start=m.get_current_plant_time(),
            timedelta=datetime.timedelta(hours=1))
    else:
        m.run_continuously(update_interval=150)      # NOTE: models may take more than 60 s to run
        # NOTE: Not able to get around the following error caused by using PySSC:
        # "AttributeError: Can't pickle local object 'CDLL.__init__.<locals>._FuncPtr'"
        # Using the pathos package instead of multiprocessing did not help.
        # Now just instead creating another Docker container for the webserver,
        # which does not hold up the continuously running Lore models.
        # update_interval = m.simulation_timestep.total_seconds()
        # p = multiprocessing.Process(target=m.run_continuously, args=(update_interval,))
        # p.start()
        # (The following adds another simultaneous mediate process, though likely not needed):
        # p = multiprocessing.Process(target=m.run_continuously, args=(1,))
        # p.start()

    return

class Mediator:
    tech_wrap = None

    def __init__(self, params_path, plant_design_path, weather_file=None, dispatch_params_path=None,
                 update_interval=datetime.timedelta(seconds=5)):
        with open(params_path) as f:
            self.params = rapidjson.load(f, parse_mode=1)
        self.weather_file = weather_file
        self.update_interval = update_interval
        self.simulation_timestep = datetime.timedelta(hours=1/self.params['time_steps_per_hour'])
        self.params['dispatch_factors_ts'] = Revenue.get_price_data(
            self.params['price_multiplier_file'],
            self.params['avg_price'],
            self.params['price_steps_per_hour'],
            self.params['time_steps_per_hour'])

        with open(plant_design_path) as f:
            plant_design = rapidjson.load(f, parse_mode=1)
        self.plant = plant_.Plant(
            design=plant_design,
            initial_state=plant_.plant_initial_state)
        
        settings.PLANT_TIME_ZONE = plant_design['timezone_string']

        parent_dir = str(Path(__file__).parents[1])
        with open(parent_dir + './data/mspt_2020_defaults.json', 'r') as f:
            ssc_params = rapidjson.load(f)
        ssc_params.update(self.plant.get_state())                       # combine default and plant params, overwriting the defaults
        ssc_params.update(self.params)                                  # combine default and mediator params, overwriting the defaults

        self.forecaster = forecasts.SolarForecast(
            self.plant.design['latitude'],
            self.plant.design['longitude'],
            self.plant.design['timezone'],  # Fixed offset!
            self.plant.design['elevation'])

        # Setup dispatch_wrap
        with open(dispatch_params_path) as f:
            dispatch_wrap_params = rapidjson.load(f, parse_mode=1)                                   # TODO: replace with a path to a JSON config file
        dispatch_wrap_params.update(self.params)                                                    # include mediator params in with dispatch_wrap_params
        self.dispatch_wrap = dispatch_wrap.DispatchWrap(plant=self.plant, params=dispatch_wrap_params)

        self.tech_wrap = tech_wrap.TechWrap(
            params=ssc_params,  # already a copy so tech_wrap cannot edit
            plant=copy.deepcopy(self.plant),  # copy so tech_wrap cannot edit
            dispatch_wrap_params=dispatch_wrap_params.copy(),  # copy so tech_wrap cannot edit
            weather_file=None)

        self.plant.update_flux_maps(self.tech_wrap.calc_flux_eta_maps(self.plant.get_design(), self.plant.get_state()))

    def run_once(self, datetime_start=None, datetime_end=None, timedelta=None):
        """Get data from external plant and weather interfaces and run entire set
        of submodels, saving data to database.

        Args:
            datetime_start: beginning of first timestep, in plant-local time
            datetime_end:   end of last timestep, in plant-local time
            timedelta:      alternative to datetime_end
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

        if datetime_start is None:
            datetime_start = self.get_current_plant_time()

        if datetime_end is None:
            if timedelta is not None:
                datetime_end = datetime_start + timedelta
            else:
                datetime_end = datetime_start + datetime.timedelta(hours=1)

        # Step 0: Normalize timesteps to even intervals and enforce timestep localization
        if datetime_start.tzinfo is None or datetime_end.tzinfo is None or \
            datetime_start.tzinfo != datetime_end.tzinfo:
            print("WARNING: Timesteps in run_once were not properly localized.")
            # Assume the datetime was given for the plant.
            _tz = pytz.FixedOffset(60 * self.plant.design['timezone'])
            datetime_start = pytz.UTC.normalize(_tz.localize(datetime_start))
            datetime_end = pytz.UTC.normalize(_tz.localize(datetime_end))
        self._validate_UTC_time(datetime_start)
        self._validate_UTC_time(datetime_end)

        datetime_start, datetime_end = normalize_timesteps(
            datetime_start,
            datetime_end,
            timestep = self.simulation_timestep.seconds / 60)

        # Step 1, Thread 1:
        # a. Call virtual/real plant to get plant operating state and any local
        #    weather data
        plant_state = self.plant.get_state()

        # b. Validate these data
        plant_state = data_validator.validate(plant_state, data_validator.plant_state_schema)

        # c. Store in database and add to current timestep cache
            #TODO: Add this

        # Step 1, Thread 2:
        # a. Get weather data and forecasts
        self.refresh_forecast_in_db(datetime_start)
        datetime_end_dispatch = datetime_start + \
            datetime.timedelta(hours=self.dispatch_wrap.params['dispatch_horizon'])
        weather_dispatch = self.get_weather_df(
            datetime_start=datetime_start,
            datetime_end=datetime_end_dispatch,
            timestep=datetime.timedelta(minutes=min(self.dispatch_wrap.params['dispatch_steplength_array'])),
            tmy3_path=self.weather_file,
            use_forecast=True)
        assert(weather_dispatch.index[0] == datetime_start + datetime.timedelta(minutes=min(self.dispatch_wrap.params['dispatch_steplength_array'])))
        assert(weather_dispatch.index[-1] == datetime_end_dispatch)

        weather_simulate = self.get_weather_df(
            datetime_start=datetime_start,
            datetime_end=datetime_end,
            timestep=datetime.timedelta(hours=1/self.params['time_steps_per_hour']),
            tmy3_path=self.weather_file,
            use_forecast=False)             # TODO: this use_forecast should probably be True too. Can we optimize these two
                                            #       get_weather_df calls since getting the forecasts takes significant time?
        assert(weather_simulate.index[0] == datetime_start + datetime.timedelta(hours=1/self.params['time_steps_per_hour']))
        assert(weather_simulate.index[-1] == datetime_end)
        self.add_weather_to_db(weather_simulate)

        # Set clearsky data
        clearsky_data = np.nan_to_num(np.array(weather_dispatch['Clear Sky DNI']), nan = 0.0)  # TODO: Better handling of nan values in clear-sky data
        clearsky_data_padded = self.tech_wrap.pad_weather_data(list_data = clearsky_data.tolist(), 
            datetime_start = self._toTMYTime(datetime_start), 
            timestep = datetime.timedelta(hours=1/self.params['time_steps_per_hour']))
        self.tech_wrap.set({'rec_clearsky_dni': clearsky_data_padded})

        # TODO(odow): keep pushing timezones through the code.

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
        # TODO: fix timezones in these db tables
        tech_outputs = {k:(list(v) if isinstance(v, tuple) else v) for (k,v) in tech_outputs.items()}   # converts tuples to lists so they can be edited
        tic = time.process_time()
        validated_outputs = data_validator.validate(tech_outputs, data_validator.ssc_schema)
        toc = time.process_time()
        print("Validation took {seconds:0.2f} seconds".format(seconds=toc-tic))
        timestamps = list(pd.date_range(
                          start=datetime_start,
                          end=datetime_end,
                          freq=self.simulation_timestep,
                          closed='right'))   # exclude start
        validated_outputs['timestamp'] = timestamps
        self.add_techdata_to_db(validated_outputs)    # TODO: rename as techmodeldata or something

        # d. Add simulated plant state and other data to cache and database, and update plant state
        new_plant_states = self.tech_wrap.get_simulated_plant_states(validated_outputs)
        # TODO: update calc_persistance_vars() so it returns for each timestep, not just end, and add to db
        new_plant_states['sf_adjust:hourly'] = self.plant.get_field_availability(
                                                          datetime_start=datetime_start,
                                                          duration=datetime_end - datetime_start,
                                                          timestep=self.simulation_timestep
                                                          )
        new_plant_states['timestamp'] = timestamps
        self.add_plantstates_to_db(new_plant_states)

        new_plant_state_persistance = self.plant.calc_persistance_vars(validated_outputs, self.simulation_timestep.seconds/3600)    # for just last timestep
        new_plant_states.update(new_plant_state_persistance)
        self.plant.set_state(new_plant_states)

        return 0

    def run_continuously(self, update_interval=150):
        """Continuously get data from external plant and weather interfaces and run
        entire set of submodels, saving data to database
        
        update_interval -- [s] how frequently the interfaces and submodels are polled and run, respectively
        """
        looping_call = LoopingCall(self.run_once)
        time.sleep(update_interval - time.time() % update_interval)          # wait to start until it's an even clock interval
        looping_call.start(update_interval)
        reactor.run()

    def get_current_plant_time(self):
        "Return the current time in UTC."
        return datetime.datetime.now(pytz.UTC)

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
    
    def _validate_UTC_time(self, time):
        "A helper function that validates the given `time` is in UTC."
        assert(time.tzinfo == pytz.UTC)
        return

    def add_weather_to_db(self, df_records):
        df_records['timestamp'] = df_records.index
        records = df_records.to_dict('records')

        instances = [
            models.WeatherData(
                timestamp =                 record['timestamp'],
                dni =                       record['DNI'],
                dhi =                       record['DHI'],
                ghi =                       record['GHI'],
                dew_point =                 record['Dew Point'],
                temperature =               record['Temperature'],
                pressure =                  record['Pressure'],
                wind_direction =            record['Wind Direction'],
                wind_speed =                record['Wind Speed'],
            )
            for record in records
        ]

        try:
            models.WeatherData.objects.bulk_create(instances, ignore_conflicts=True)
            # If ignore_conflicts=False and if any to-be-added records are already in the database, as indicated by the timestamp,
            #  an exception is raised and no to-be-added records are added.
            # If ignore_conflicts=True, all records not already in the database are added. To-be-added records that are already in the
            #  database do not replace the database records. Therefore, no existing database records are overwritten.
        except IntegrityError as err:
            error_string = format(err)
            if error_string == "UNIQUE constraint failed: mediation_weatherdata.timestamp":
                raise IntegrityError(error_string)      # just re-raise the exception for now
        except Exception as err:
            raise(err)

    def add_plantstates_to_db(self, records):
        n_records = len(records['timestamp'])

        instances = [
            models.PlantStateData(
                timestamp =                 records['timestamp'][i],
                is_field_tracking =         records['is_field_tracking_init'][i],
                receiver_mode =             records['rec_op_mode_initial'][i],
                dt_rec_startup_remain =     records['rec_startup_time_remain_init'][i],
                dE_rec_startup_remain =     records['rec_startup_energy_remain_init'][i] * 1.e-3,
                # dt_rec_current_mode =       records['disp_rec_persist0'][i],
                # dt_rec_not_on =             records['disp_rec_off0'][i],
                sf_adjust =                 records['sf_adjust:hourly'][i],
                T_cold_tank =               records['T_tank_cold_init'][i],
                T_hot_tank =                records['T_tank_hot_init'][i],
                Frac_avail_hot_tank =       records['csp_pt_tes_init_hot_htf_percent'][i],
                cycle_mode =                records['pc_op_mode_initial'][i],
                dt_cycle_startup_remain =   records['pc_startup_time_remain_init'][i],
                dE_cycle_startup_remain =   records['pc_startup_energy_remain_initial'][i],
                # dt_cycle_current_mode =     records['disp_pc_persist0'][i],
                # dt_cycle_not_on =           records['disp_pc_off0'][i],
                W_cycle =                   records['wdot0'][i] * 1.e3,
                Q_cycle =                   records['qdot0'][i] * 1.e3,
            )
            for i in range(n_records)
        ]

        try:
            models.PlantStateData.objects.bulk_create(instances, ignore_conflicts=True)
            # If ignore_conflicts=False and if any to-be-added records are already in the database, as indicated by the timestamp,
            #  an exception is raised and no to-be-added records are added.
            # If ignore_conflicts=True, all records not already in the database are added. To-be-added records that are already in the
            #  database do not replace the database records. Therefore, no existing database records are overwritten.
        except IntegrityError as err:
            error_string = format(err)
            if error_string == "UNIQUE constraint failed: mediation_plantstates.timestamp":
                raise IntegrityError(error_string)      # just re-raise the exception for now
        except Exception as err:
            raise(err)

    def add_techdata_to_db(self, records):
        n_records = len(records['timestamp'])

        instances = [
            models.TechData(
                timestamp =             records['timestamp'][i],
                E_tes_charged =         records['e_ch_tes'][i] * 1.e3,
                eta_tower_thermal =     records['eta_therm'][i],
                eta_field_optical =     records['eta_field'][i],
                W_grid_no_derate =      records['P_out_net'][i] * 1.e3,
                tou =                   records['tou_value'][i],
                W_grid_with_derate =    records['gen'][i],
                Q_tower_incident =      records['q_dot_rec_inc'][i] * 1.e3,
                Q_field_incident =      records['q_sf_inc'][i] * 1.e3,
                pricing_multiple =      records['pricing_mult'][i],
                dni =                   records['beam'][i],
                Q_tower_absorbed =      records['Q_thermal'][i] * 1.e3,
                mdot_tower =            records['m_dot_rec'][i],
                mdot_cycle =            records['m_dot_pc'][i],
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

    def refresh_forecast_in_db(self, datetime_start):
        """
        Update the database with the latest forecast.
        """
        data = self.forecaster.get_raw_data(datetime_start)
        instances = [
            models.SolarForecastData(
                timestamp=pytz.UTC.normalize(time),
                # Choice: use DNI directly from pvlib, or our forecast?
                # In data from the site, we observed a bias in the NDFD data 
                # that underestimates actual DNI. The forecast corrects that.
                # dni=row.dni,   # pvlib
                dni=row['0.5'],  # our forecast
                dhi=row.dhi,
                ghi=row.ghi,
                temperature=row.temp_air,
                pressure=row.pressure,
                wind_speed=row.wind_speed,
                clear_sky=row.clear_sky,
                ratio=row.ratio,
                dni_10=row['0.1'],
                dni_25=row['0.25'],
                dni_50=row['0.5'],
                dni_75=row['0.75'],
                dni_90=row['0.9'],
            )
            # Okay, I know data.iterrows is slow. But the dataframe is never
            # very big.
            for (time, row) in data.iterrows()
        ]
        models.SolarForecastData.objects.bulk_create(
            instances, 
            ignore_conflicts=True,
        )
        return

    def get_forecast(
        self,
        index,
        resolution=pd.Timedelta(hours=1),
    ):
        """
        Return the most recent DNI forecast issued before `datetime_start`,
        covering the time between `datetime_start` and `datetime_end`.

        Parameters
        ----------
        datetime_start : timezone aware datetime
            Start of the forecast window.
        resolution : pd.Timedelta
            The resolution passed to `pd.resample` for resampling the
            NDFD forecast into finer resolution. Defaults to `hours = 1`.
        horizon : pd.Timedelta
            Length of the forecast window.
        """
        query = models.SolarForecastData.objects.filter(
            timestamp__gte=index[0] - datetime.timedelta(hours=1)
        )
        raw_data = pd.DataFrame(query.values())
        raw_data.set_index('timestamp', inplace=True)
        data = raw_data.resample(resolution).mean()
        # Clean up the ratio by imputing any NaNs that arose to the nearest
        # non-NaN value. This means we assume that the start of the day acts
        # like the earliest observation, and the end of the day looks like the
        # last observation.
        data.interpolate(method='linear', inplace=True)
        # However, nearest only works when there are non-NaN values either side.
        # For the first and last NaNs, use bfill and ffill:
        data.fillna(method='bfill', inplace=True)
        data.fillna(method='ffill', inplace=True)
        data = data[data.index >= index[0]]
        data = data[data.index <= index[-1]]
        return data
        
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
        `datetime_end`, which are in plant-local time.

        Weather files give data at the middle of the timestep (as they are integrated
        average values) and the models expect that datetimes designate the end of the
        timestep; however, they also expect that the start datetime is at the
        beginning of the first timestep. Therefore, datetime_start is advanced by one
        timestep before passing to the weather data getter, so it designates the end
        of the first timestep.

        If `use_forecast`, replace the 'DNI', 'DHI', 'GHI', 'Temperature' and
        'Wind Speed' columns with the latest NDFD forecast from the forecasts
        submodule. In addition, create a new column: 'Clear Sky DNI'.
        """
        datetime_start += timestep      # converting to end of first timestep, by convention
        self._validate_UTC_time(datetime_start)
        self._validate_UTC_time(datetime_end)
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
        data.index = data.index.tz_localize(tmy_tz).tz_convert(pytz.UTC)
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
        data.fillna(method='ffill', inplace=True)
        # Now strip the data back to what the user asked for:
        data = data[(data.index >= datetime_start) & (data.index <= datetime_end)]
        if use_forecast:
            tic = time.process_time()
            solar_forecast = self.get_forecast(data.index, resolution=timestep)
            toc = time.process_time()
            print("Generating forecast took {seconds:.2f} seconds".format(seconds=toc-tic))
            key_map = {
                'dni': 'DNI',
                'dhi': 'DHI',
                'ghi': 'GHI',
                'wind_speed': 'Wind Speed',
                'temperature': 'Temperature',
                # This is not part of the TMY file!
                'clear_sky': 'Clear Sky DNI',
                'pressure': 'Pressure',
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
    new_end = round_minutes(datetime_end, 'down', timestep)
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


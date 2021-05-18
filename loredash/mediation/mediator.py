from django.conf import settings
from django.db import IntegrityError
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import time, copy, datetime, math
import numpy as np
from twisted.internet.task import LoopingCall
from twisted.internet import reactor
import PySAM_DAOTk.TcsmoltenSalt as pysam
from pathlib import Path
import pandas as pd
import rapidjson
from mediation import data_validator, pysam_wrap, dispatch_wrap, models
import mediation.plant as plant_
import librtdispatch.util as util
from mediation.plant import Revenue
from data.mspt_2020_defaults import default_ssc_params
# import models

class Mediator:
    pysam_wrap = None
    validated_outputs_prev = None

    def __init__(self, params, plant_config_path, plant_design, weather_file=None,
                 update_interval=datetime.timedelta(seconds=5), start_date_year=2018):
        self.params = params
        self.weather_file = weather_file
        self.update_interval = update_interval
        self.simulation_timestep = datetime.timedelta(hours=1/self.params['time_steps_per_hour'])

        self.plant = plant_.Plant(
            design=plant_design,
            initial_state=plant_.plant_initial_state,
            )
        self.params['dispatch_factors_ts'] = Revenue.get_price_data(self.params['price_multiplier_file'], self.params['avg_price'],
            self.params['price_steps_per_hour'], self.params['time_steps_per_hour'])

        default_ssc_params.update(self.params)                                  # combine default and mediator params, overwriting the defaults
        self.pysam_wrap = pysam_wrap.PysamWrap(
            mediator_params=default_ssc_params,                                 # already a copy so pysam_wrap cannot edit
            plant=copy.deepcopy(self.plant),                                    # copy so pysam_wrap cannot edit
            dispatch_wrap_params=dispatch_wrap.dispatch_wrap_params.copy(),     # copy so pysam_wrap cannot edit
            weather_file=None
            )

        #TODO: Add a function to pysam_wrap that creates a generic weather_dataframe and don't pass a weather_dataframe here
        weather_dataframe = self.GetWeatherDataframe(
            datetime.datetime(start_date_year, 1, 1, 0),
            datetime.datetime(start_date_year, 12, 31, 23),
            tmy3_path=self.weather_file
            )

        #TODO: Where should Forecasts be initialized?
        # Add an equivalent call of the following to Forecasts. Need this set for the later calc_flux_eta_maps()
        timestep = weather_dataframe.index[1] - weather_dataframe.index[0]
        clearsky_data = get_clearsky_data(
            clearsky_file=self.params['clearsky_file'],
            datetime_start=weather_dataframe.index[0],
            duration=weather_dataframe.index[-1] - weather_dataframe.index[0] + timestep,   # need to add timestep, e.g., 0:00 to 1:00 is two hours, not one
            timestep=timestep)
        self.pysam_wrap.ssc.set({'rec_clearsky_dni': clearsky_data})        #TODO: use a new pysam_wrap set function instead of reaching in to ssc.set()

        
        self.plant.update_design(self.pysam_wrap.calc_flux_eta_maps(weather_dataframe=weather_dataframe))

        # Setup dispatch_wrap
        # sf_adjust_hourly = util.get_field_availability_adjustment(self.params['time_steps_per_hour'], start_date_year, self.params['control_field'],
        #     self.params['use_CD_measured_reflectivity'], self.plant.design, self.params['fixed_soiling_loss'])
        # ground_truth_weather_data = util.get_ground_truth_weather_data(self.params['ground_truth_weather_file'], self.params['time_steps_per_hour'])
        # dispatch_wrap_data = {
            # 'sf_adjust:hourly':                 sf_adjust_hourly,
            # 'dispatch_factors_ts':              price_data,
            # 'solar_resource_data':              ground_truth_weather_data,
            # 'clearsky_data':                    clearsky_data,
        # }
        dispatch_wrap_params = dispatch_wrap.dispatch_wrap_params
        dispatch_wrap_params['clearsky_data'] = util.get_clearsky_data(self.params['clearsky_file'], self.params['time_steps_per_hour']) # legacy call. TODO: cleanup?
        dispatch_wrap_params.update(self.params)                                                    # include mediator params in with dispatch_wrap_params
        dispatch_wrap_params['start_date'] = datetime.datetime(start_date_year, 1, 1, 8, 0, 0)      # needed for schedules TODO: fix spanning years (note the 8)
        self.dispatch_wrap = dispatch_wrap.DispatchWrap(plant=self.plant, params=dispatch_wrap.dispatch_wrap_params)
        self.dispatch_inputs = {
            'ursd_last':                        None,
            'yrsd_last':                        None,
            'weather_data_for_dispatch':        None,
            'current_day_schedule':             None,
            'next_day_schedule':                None,
            'current_forecast_weather_data':    None,
            'schedules':                        None,
            'horizon':                          1/self.params['time_steps_per_hour']*3600,          # [s]
        }

    
    def RunOnce(self, datetime_start=None, datetime_end=None):
        """
        Get data from external plant and weather interfaces and run
        entire set of submodels, saving data to database

        datetime_start = beginning of first timestep
        datetime_end = end of last timestep

        if datetime_start = none, the timestep including the current time will be run
        e.g., if current clock time is 17:43 and the simulation_timestep = 5 minutes,
        the timestep from 17:40 to 17:45 will be run, meaning:
            datetime_start = 17:40
            datetime_end = 17:45
        """

        # The planned code:
        # Step 1:
        #   Thread 1:
        #       a. If virtual plant, query database for needed inputs (or use cache from previous timestep)
        #       b. Call virtual/real plant to get plant operating state and any local weather data
        #       c. Validate these data
        #       d. Store in database and add to current timestep cache
        #   Thread 2:
        #       a. Call module to retrieve weather data(s)
        #       b. Validate these data
        #       c. Store in database and add to current timestep cache
        #
        # Step 2:
        #   Thread 1:
        #       a. Get dispatch model inputs from data cache of current and/or previous timestep
        #       b. Call dispatch model using inputs
        #       c. Validate these data
        #       d. Store in database and add to current timestep cache
        #
        # Step 3:
        #   Thread 1:
        #       a. Set plant state and weather values in PySAM
        #       b. Call PySAM using inputs
        #       c. Validate these data
        #       d. Add data to cache and store in database
        #
        # Step 4:
        #   Thread 1:
        #       a. Update previous timestep cache with current timestep cache and then clear current cache


        # Step 3, Thread 1:
        # Normalize timesteps to even intervals, even if they are given
        datetime_now = datetime.datetime.now()
        if isinstance(datetime_start, datetime.datetime):
            datetime_start = RoundMinutes(datetime_start, 'down', self.simulation_timestep.seconds/60)
            if isinstance(datetime_end, datetime.datetime):
                datetime_end = RoundMinutes(datetime_end, 'up', self.simulation_timestep.seconds/60)
            else:
                datetime_end = datetime_start + self.simulation_timestep
        else:
            datetime_start = RoundMinutes(datetime_now, 'down', self.simulation_timestep.seconds/60)    # the start of the time interval currently in
            datetime_end = datetime_start + self.simulation_timestep        # disregard a given datetime_end if there is no given datetime_start
        
        print("Datetime now = {datetime}".format(datetime=datetime_now))
        print("Start datetime = {datetime}".format(datetime=datetime_start))
        print("End datetime = {datetime}".format(datetime=datetime_end))

        # a. Set weather values and plant state for PySAM
        weather_dataframe = self.GetWeatherDataframe(datetime_start, datetime_end, tmy3_path=self.weather_file)
        
        # b. Call dispatch model, (which includes a PySAM model run to get estimates) and update inputs for next call
        dispatch_outputs = self.dispatch_wrap.run(
            start_date=datetime_start,
            timestep_days=self.simulation_timestep.seconds/(24*3600),
            horizon=self.dispatch_inputs['horizon'],
            retvars=default_disp_stored_vars(),
            ursd_last=self.dispatch_inputs['ursd_last'],
            yrsd_last=self.dispatch_inputs['yrsd_last'],
            current_forecast_weather_data=self.dispatch_inputs['current_forecast_weather_data'],
            weather_data_for_dispatch=self.dispatch_inputs['weather_data_for_dispatch'],
            schedules=self.dispatch_inputs['schedules'],
            current_day_schedule=self.dispatch_inputs['current_day_schedule'],
            next_day_schedule=self.dispatch_inputs['next_day_schedule'],
            f_estimates_for_dispatch_model=self.pysam_wrap.estimates_for_dispatch_model,
            initial_plant_state=self.plant.state
        )
        self.dispatch_inputs['ursd_last'] = dispatch_outputs['ursd_last']
        self.dispatch_inputs['yrsd_last'] = dispatch_outputs['yrsd_last']
        self.dispatch_inputs['weather_data_for_dispatch'] = dispatch_outputs['weather_data_for_dispatch']
        self.dispatch_inputs['current_day_schedule'] = dispatch_outputs['current_day_schedule']
        self.dispatch_inputs['next_day_schedule'] = dispatch_outputs['next_day_schedule']
        self.dispatch_inputs['current_forecast_weather_data'] = dispatch_outputs['current_forecast_weather_data']
        self.dispatch_inputs['schedules'] = dispatch_outputs['schedules']
        self.dispatch_inputs['horizon'] -= int(self.simulation_timestep.seconds)

        # Step 1, Thread 2:?
        # a. Call Forecasts
        #TODO: add this call to forecasts.py
        clearsky_data = get_clearsky_data(
            clearsky_file=self.params['clearsky_file'],
            datetime_start=datetime_start,
            duration=datetime_start - datetime_end,
            timestep=self.simulation_timestep)
        self.pysam_wrap.ssc.set({'rec_clearsky_dni': clearsky_data})        #TODO: use a new pysam_wrap set function.


        # b. Call PySAM
        self.pysam_wrap.ssc.set(dispatch_outputs['ssc_dispatch_targets'])       #TODO: make pysam_wrap function for set (don't just reach in and use ssc.set())

        tech_outputs = self.pysam_wrap.Simulate(datetime_start, datetime_end, self.simulation_timestep, self.plant.get_state(), weather_dataframe=weather_dataframe)
        print("Annual Energy [kWh]= ", tech_outputs["annual_energy"])

        new_plant_state_vars = self.pysam_wrap.GetSimulatedPlantState(tech_outputs)      # for initializing next simulation from a prior one
        self.plant.update_state(tech_outputs, new_plant_state_vars, self.simulation_timestep.seconds/3600)

        # c. Validate these data
        # wanted_keys = ['time_hr', 'e_ch_tes', 'eta_therm', 'eta_field', 'P_out_net', 'tou_value', 'gen', 'q_dot_rec_inc', 'q_sf_inc', 'pricing_mult', 'beam']
        # wanted_keys = list(set(tech_outputs.keys()))       # all keys are wanted
        # wanted_outputs = dict((k, tech_outputs[k]) for k in wanted_keys if k in tech_outputs)
        wanted_outputs = tech_outputs
        wanted_outputs = {k:(list(v) if isinstance(v, tuple) else v) for (k,v) in wanted_outputs.items()}     # converts tuples to lists so they can be edited
        tic = time.process_time()
        validated_outputs = data_validator.validate(wanted_outputs, data_validator.pysam_schema)
        toc = time.process_time()
        print("Validation took {seconds:0.2f} seconds".format(seconds=toc-tic))
        validated_outputs['year_start'] = datetime_start.year       # add this so date can be determined from time_hr

        # d. Add data to cache and store in database
        self.validated_outputs_prev = copy.deepcopy(validated_outputs)
        self.BulkAddToPysamTable(validated_outputs)

        return 0

    def RunContinuously(self, update_interval=5):
        """Continuously get data from external plant and weather interfaces and run
        entire set of submodels, saving data to database
        
        update_interval -- [s] how frequently the interfaces and submodels are polled and run, respectively
        """
        looping_call = LoopingCall(self.RunOnce)
        time.sleep(update_interval - time.time() % update_interval)          # wait to start until it's an even clock interval
        looping_call.start(update_interval)
        reactor.run()

    def ModelPreviousDayAndAddToDb(self):
        """
        Simulate previous day and add to database
        e.g.:
        if current time is 17:43 and simulation_timestep = 5 minutes:
            it will model from 17:40 yesterday (start of timestep)
            to 17:40 today (end of timestep)
            with timesteps in database (end of timesteps) being from 17:45 yesterday to 17:40 today
            for 288 total new entries
        """
        datetime_now = datetime.datetime.now()
        datetime_now_rounded_down = RoundMinutes(datetime_now, 'down', self.simulation_timestep.seconds/60)    # the start of the time interval currently in
        datetime_start_prev_day = datetime_now_rounded_down - datetime.timedelta(days=1)
        datetime_end_current_day = datetime_now_rounded_down                   # end of the last timestep
                                                                            # (as noted for "time_stop" on line 1004 in cmod_tcsmolten_salt.cpp)
        self.RunOnce(datetime_start_prev_day, datetime_end_current_day)
        return 0

    def BulkAddToPysamTable(self, records):
        n_records = len(records['time_hr'])
        newyears = datetime.datetime(records['year_start'], 1, 1, 0, 0, 0)

        instances = [
            models.PysamData(
                timestamp =             RoundTime(newyears + datetime.timedelta(hours=records['time_hr'][i]), 1),       # round to nearest second
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
            models.PysamData.objects.bulk_create(instances, ignore_conflicts=True)
            # If ignore_conflicts=False and if any to-be-added records are already in the database, as indicated by the timestamp,
            #  an exception is raised and no to-be-added records are added.
            # If ignore_conflicts=True, all records not already in the database are added. To-be-added records that are already in the
            #  database do not replace the database records. Therefore, no existing database records are overwritten.
        except IntegrityError as err:
            error_string = format(err)
            if error_string == "UNIQUE constraint failed: mediation_pysamdata.timestamp":
                raise IntegrityError(error_string)      # just re-raise the exception for now
        except Exception as err:
            raise(err)
    
    def GetWeatherDataframe(self, datetime_start, datetime_end, **kwargs):
        """put the weather forecast call here instead"""
        tmy3_path = kwargs.get('tmy3_path') if 'tmy3_path' in kwargs else None
        return Tmy3ToDataframe(tmy3_path, datetime_start, datetime_end)

class Plant:
    @staticmethod
    def LoadPlantConfig(config_path):
        if isinstance(config_path, str) and os.path.isfile(config_path):
            with open(config_path) as f:
                plant_config = rapidjson.load(f)
        else:
            raise Exception('Plant configuration file not found.')
            
        validated_outputs = data_validator.validate(plant_config, data_validator.plant_config_schema)

        plant_config_table = PlantConfig()
        plant_config_table.name = plant_config['name']
        plant_config_table.save()
        Plant.LoadPlantLocation(plant_config['location'], validate=False)    # already validated above
        del plant_config

    @staticmethod
    def LoadPlantLocation(plant_location, validate=True):
        if validate == True:
            plant_location = data_validator.validate(plant_location, data_validator.plant_location_schema)

        plant_config_table = PlantConfig()
        plant_config_table.latitude = plant_location['latitude']
        plant_config_table.longitude = plant_location['longitude']
        plant_config_table.elevation = plant_location['elevation']
        plant_config_table.timezone = plant_location['timezone']
        plant_config_table.timezone_string = plant_location['timezone_string']
        plant_config_table.save()

    @staticmethod
    def GetPlantConfig():
        result = list(PlantConfig.objects.filter(site_id=settings.SITE_ID).values())[0]
        result['location'] = {}
        result['location']['latitude'] = result.pop('latitude')
        result['location']['longitude'] = result.pop('longitude')
        result['location']['elevation'] = result.pop('elevation')
        result['location']['timezone'] = result.pop('timezone')
        result['location']['timezone_string'] = result.pop('timezone_string')
        return result

    @staticmethod
    def GetPlantState(validated_outputs_prev, **kwargs):
        """put virtual/real call here instead"""
        assert 'pysam_wrap' in kwargs
        pysam_wrap = kwargs.get('pysam_wrap')
        plant_state = pysam_wrap.GetSimulatedPlantState(validated_outputs_prev)      # for initializing next simulation from a prior one
        if plant_state is None:
            plant_state = pysam_wrap.GetDefaultPlantState()
        return plant_state    

def MediateContinuously(update_interval=5):
    mediator = Mediator()
    mediator.RunContinuously(update_interval=update_interval)
    return False

# def MediateOnce():
#     """This will likely only be used for testing"""
#     mediator = Mediator()
#     mediator.RunOnce()
#     return False

def RoundTime(dt, second_resolution):
    """Round to nearest second interval"""
    seconds = (dt.replace(tzinfo=None) - dt.min).seconds + dt.microsecond * 1.e-6
    rounding = (seconds+second_resolution/2) // second_resolution * second_resolution
    return dt + datetime.timedelta(0,rounding-seconds,0)

def RoundMinutes(dt, direction, minute_resolution):
    """
    Round to nearest minute interval
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

def Tmy3ToDataframe(tmy3_path, datetime_start=None, datetime_end=None):
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
    location = GetLocationFromWeatherFile(tmy3_path)
    df.attrs.update(location)

    # Ensure the requested datetimes are contained within the returned weather datetimes
    df_timestep = df.index[1] - df.index[0]
    datetime_start_adj = RoundMinutes(datetime_start, 'down', df_timestep.seconds/60)
    datetime_end_adj = RoundMinutes(datetime_end, 'up', df_timestep.seconds/60)
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

def GetLocationFromWeatherFile(tmy3_path):
    df_meta = pd.read_csv(tmy3_path, sep=',', header=0, nrows=1)
    return {
        'latitude': float(df_meta['Latitude'][0]),
        'longitude': float(df_meta['Longitude'][0]),
        'timezone': int(df_meta['Time Zone'][0]),
        'elevation': float(df_meta['Elevation'][0])
    }


#TODO: Remove this function and replace with one in forecasts.py
def get_clearsky_data(clearsky_file, datetime_start=None, duration=None, timestep=None):
    """output array must be equal in length to the weather data to satisfy ssc"""
    if datetime_start is None:
        datetime_start = datetime.datetime(2018, 1, 1)
    if duration is None:
        duration = datetime.timedelta(days=365)
    if timestep is None:
        timestep = datetime.timedelta(minutes=1)

    CLEARSKY_DAYS_GENERATED = 365
    steps_per_hour = 1/(timestep.total_seconds()/3600)
    clearsky_data = util.get_clearsky_data(clearsky_file, steps_per_hour).tolist()
    assert(len(clearsky_data) == steps_per_hour * 24 * CLEARSKY_DAYS_GENERATED)
    df = pd.DataFrame(clearsky_data, columns=['clearsky_data'])
    df.index = pd.date_range(start=datetime_start,
                             end=datetime_start + datetime.timedelta(days=CLEARSKY_DAYS_GENERATED) - timestep,
                             freq=timestep)

    df_out = df[datetime_start:(datetime_start + duration - timestep)]
    return list(df_out['clearsky_data'])

# This is duplicated in case_study.py
mediator_params = {
    # Control conditions
	'time_steps_per_hour':			        60,			            # Simulation time resolution in ssc (1min)   DUPLICATED to: ssc_time_steps_per_hour
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
    'price_multiplier_file':                '../../librtdispatch/prices_flat.csv',
	'ppa_multiplier_model':			        1,
	'price_steps_per_hour':			        1,                      # Number of steps per hour in electricity price multipliers
	'avg_price':					        138,                    # Average electricity price ($/MWh):  CD original PPA was $138/MWh
    'avg_purchase_price':                   30,                     # Average electricity purchase price ($/MWh) (note, this isn't currently used in the dispatch model)
    'avg_price_disp_storage_incentive':     0.0,                    # Average electricity price ($/MWh) used in dispatch model storage inventory incentive

    # Field, receiver, and cycle simulation options
    'clearsky_file':                        './model-validation/input_files/weather_files/clearsky_pvlib_ineichen_1min_2018.csv',   # Expected clear-sky DNI from Ineichen model (via pvlib). 
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


def default_ssc_return_vars():
    return ['beam', 'clearsky', 'tdry', 'wspd', 'solzen', 'solaz', 'pricing_mult',
                'sf_adjust_out', 'q_sf_inc', 'eta_field', 'defocus', 
                'q_dot_rec_inc', 'Q_thermal', 'm_dot_rec', 'q_startup', 'q_piping_losses', 'q_thermal_loss', 'eta_therm',
                'T_rec_in', 'T_rec_out', 'T_panel_out',
                'T_tes_hot', 'T_tes_cold', 'mass_tes_cold', 'mass_tes_hot', 'q_dc_tes', 'q_ch_tes', 'e_ch_tes', 'tank_losses', 'hot_tank_htf_percent_final',
                'm_dot_cr_to_tes_hot', 'm_dot_tes_hot_out', 'm_dot_pc_to_tes_cold', 'm_dot_tes_cold_out', 'm_dot_field_to_cycle', 'm_dot_cycle_to_field',
                'P_cycle','eta', 'T_pc_in', 'T_pc_out', 'q_pb', 'q_dot_pc_startup', 'P_out_net', 
                'P_tower_pump', 'htf_pump_power', 'P_cooling_tower_tot', 'P_fixed', 'P_plant_balance_tot', 'P_rec_heattrace', 'q_heater', 'P_cycle_off_heat', 'pparasi',
                'is_rec_su_allowed', 'is_pc_su_allowed', 'is_pc_sb_allowed',
                'op_mode_1', 'op_mode_2', 'op_mode_3', 'q_dot_est_cr_on', 'q_dot_est_cr_su', 'q_dot_est_tes_dc', 'q_dot_est_tes_ch', 'q_dot_pc_target_on'
                ]


def default_disp_stored_vars():
    return ['cycle_on', 'cycle_standby', 'cycle_startup', 'receiver_on', 'receiver_startup', 'receiver_standby', 
            'receiver_power', 'thermal_input_to_cycle', 'electrical_output_from_cycle', 'net_electrical_output', 'tes_soc',
            'yrsd', 'ursd']

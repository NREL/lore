from django.conf import settings
from django.db import IntegrityError
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import time, copy, datetime, math
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
# import models

class Mediator:
    pysam_wrap = None
    validated_outputs_prev = None
    default_pysam_model = "MSPTSingleOwner"

    def __init__(self, params, plant_design, override_with_weather_file_location=False,
                 weather_file=None, preprocess_pysam=True, preprocess_pysam_on_init=True,
                 update_interval=datetime.timedelta(seconds=5), start_date_year=2018):
        self.params = params
        self.override_with_weather_file_location = override_with_weather_file_location
        self.weather_file = weather_file
        self.preprocess_pysam = preprocess_pysam
        self.preprocess_pysam_on_init = preprocess_pysam_on_init
        self.update_interval = update_interval
        self.simulation_timestep = datetime.timedelta(hours=1/self.params['time_steps_per_hour'])

        # TODO: reinstitute this validation.
        # if plant_config_path is None:
        #     # Verify plant configuration in database
        #     try:
        #         data_validator.validate(Plant.GetPlantConfig(), data_validator.plant_config_schema)
        #     except Exception as err:
        #         raise(err)  # just re-raise for now

        if self.params['control_receiver'] == 'CD_data':
            plant_design['rec_user_mflow_path_1'], plant_design['rec_user_mflow_path_2'] = get_user_flow_paths(
                flow_path1_file=self.params['CD_mflow_path2_file'],          # Note ssc path numbers are reversed relative to CD path numbers
                flow_path2_file=self.params['CD_mflow_path1_file'],
                time_steps_per_hour=self.params['time_steps_per_hour'],
                helio_reflectance=plant_design['helio_reflectance'],
                use_measured_reflectivity=self.params['use_CD_measured_reflectivity'],
                soiling_avail=util.get_CD_soiling_availability(start_date_year, plant_design['helio_reflectance'] * 100), # CD soiled / clean reflectivity (daily array),
                fixed_soiling_loss=self.params['fixed_soiling_loss']
                )
            plant_design['rec_user_mflow_path_1'] = rec_user_mflow_path_1
            plant_design['rec_user_mflow_path_2'] = rec_user_mflow_path_2
        else:
            rec_user_mflow_path_1 = None
            rec_user_mflow_path_2 = None

        plant_design['is_elec_heat_dur_off'] = [plant_design['is_elec_heat_dur_off']]       # NOTE: made this value a list as needed by PySAM, but needs to not be one for PySSC

        self.plant = plant_.Plant(
            design=plant_design,
            initial_state=plant_.plant_initial_state
            )

        if weather_file is not None and override_with_weather_file_location == True:
            self.plant.set_location(GetLocationFromWeatherFile(weather_file))

        self.pysam_wrap = pysam_wrap.PysamWrap(
            mediator_params=self.params.copy(),
            plant=self.plant,
            dispatch_wrap_params=dispatch_wrap.dispatch_wrap_params.copy(),
            model_name=self.default_pysam_model,
            load_defaults=False,
            weather_file=None,
            enable_preprocessing=self.preprocess_pysam,
            preprocess_on_init=self.preprocess_pysam_on_init
            )

        # Setup dispatch_wrap
        sf_adjust_hourly = util.get_field_availability_adjustment(self.params['time_steps_per_hour'], start_date_year, self.params['control_field'],
            self.params['use_CD_measured_reflectivity'], self.plant.design, self.params['fixed_soiling_loss'])
        price_data = Revenue.get_price_data(self.params['price_multiplier_file'], self.params['avg_price'], self.params['price_steps_per_hour'], self.params['time_steps_per_hour'])
        clearsky_data = util.get_clearsky_data(self.params['clearsky_file'], self.params['time_steps_per_hour'])
        ground_truth_weather_data = util.get_ground_truth_weather_data(self.params['ground_truth_weather_file'], self.params['time_steps_per_hour'])
        dispatch_wrap_data = {
            'sf_adjust:hourly':                 sf_adjust_hourly,
            'dispatch_factors_ts':              price_data,
            'clearsky_data':                    clearsky_data,
            'solar_resource_data':              ground_truth_weather_data,
            'rec_user_mflow_path_1':            rec_user_mflow_path_1,
            'rec_user_mflow_path_2':            rec_user_mflow_path_2,
        }
        dispatch_wrap_params = dispatch_wrap.dispatch_wrap_params
        dispatch_wrap_params.update(self.params)                                                    # include mediator params in with dispatch_wrap_params
        dispatch_wrap_params['start_date'] = datetime.datetime(start_date_year, 1, 1, 8, 0, 0)      # needed for schedules TODO: fix spanning years (note the 8)
        self.dispatch_wrap = dispatch_wrap.DispatchWrap(plant=self.plant, params=dispatch_wrap.dispatch_wrap_params, data=dispatch_wrap_data)
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


        # b. Call PySAM
        self.pysam_wrap._SetTechModelParams(
            dispatch_outputs['ssc_dispatch_targets'].target_for_pysamwrap()
        )
        tech_outputs = self.pysam_wrap.Simulate(datetime_start, datetime_end, self.simulation_timestep, self.plant.state, weather_dataframe=weather_dataframe)
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

    if timestamp_end_query > timestamp_start_query:     # if dates aren't in different years
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


# This is duplicated in case_study.py
mediator_params = {
    # Control conditions
	'time_steps_per_hour':			        60,			            # Simulation time resolution in ssc (1min)   DUPLICATED to: ssc_time_steps_per_hour
	'is_dispatch':					        0,                      # Always disable dispatch optimization in ssc
	'is_dispatch_targets':			        True,		            # True if (is_optimize or control_cycle == 'CD_data')
    'is_optimize':					        True,                   # Use dispatch optimization
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
    'ground_truth_weather_file':            './model-validation/input_files/weather_files/ssc_weatherfile_1min_2018.csv',   # Weather file derived from CD data: DNI, ambient temperature,
                                                                                                                            #  wind speed, etc. are averaged over 4 CD weather stations,
                                                                                                                            #  after filtering DNI readings for bad measurements. 
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


def reupdate_ssc_constants(D, params, data):
    D['solar_resource_data'] = data['solar_resource_data']
    D['dispatch_factors_ts'] = data['dispatch_factors_ts']

    D['ppa_multiplier_model'] = params['ppa_multiplier_model']
    D['time_steps_per_hour'] = params['time_steps_per_hour']
    D['is_rec_model_trans'] = params['is_rec_model_trans']
    D['is_rec_startup_trans'] = params['is_rec_startup_trans']
    D['rec_control_per_path'] = params['rec_control_per_path']
    D['field_model_type'] = params['field_model_type']
    D['eta_map_aod_format'] = params['eta_map_aod_format']
    D['is_rec_to_coldtank_allowed'] = params['is_rec_to_coldtank_allowed']
    D['is_dispatch'] = params['is_dispatch']
    D['is_dispatch_targets'] = params['is_dispatch_targets']

    #--- Set field control parameters
    if params['control_field'] == 'CD_data':
        D['rec_su_delay'] = params['rec_su_delay']
        D['rec_qf_delay'] = params['rec_qf_delay']

    #--- Set receiver control parameters
    if params['control_receiver'] == 'CD_data':
        D['is_rec_user_mflow'] = params['is_rec_user_mflow']
        D['rec_su_delay'] = params['rec_su_delay']
        D['rec_qf_delay'] = params['rec_qf_delay']
    elif params['control_receiver'] == 'ssc_clearsky':
        D['rec_clearsky_fraction'] = params['rec_clearsky_fraction']
        D['rec_clearsky_model'] = params['rec_clearsky_model']
        D['rec_clearsky_dni'] = data['clearsky_data'].tolist()
    elif params['control_receiver'] == 'ssc_actual_dni':
        D['rec_clearsky_fraction'] = params['rec_clearsky_fraction']

    return


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


def get_user_flow_paths(flow_path1_file, flow_path2_file, time_steps_per_hour, helio_reflectance, use_measured_reflectivity,
    soiling_avail=None, fixed_soiling_loss=None):
    """Load user flow paths into the ssc input dict (D)"""
    flow_path1_data = np.genfromtxt(flow_path1_file)
    flow_path2_data = np.genfromtxt(flow_path2_file)
    if time_steps_per_hour != 60:
        flow_path1_data = np.array(util.translate_to_new_timestep(flow_path1_data, 1./60, 1./time_steps_per_hour))
        flow_path2_data = np.array(util.translate_to_new_timestep(flow_path2_data, 1./60, 1./time_steps_per_hour))
    mult = np.ones_like(flow_path1_data)
    if not use_measured_reflectivity:                                                   # Scale mass flow based on simulated reflectivity vs. CD actual reflectivity
        rho = (1-fixed_soiling_loss) * helio_reflectance                                # Simulated heliostat reflectivity
        mult = rho*np.ones(365) / (soiling_avail*helio_reflectance)                     # Ratio of simulated / CD reflectivity
        mult = np.repeat(mult, 24*time_steps_per_hour)                                  # Annual array at ssc resolution
    rec_user_mflow_path_1 = (flow_path1_data * mult).tolist()                          # Note ssc path numbers are reversed relative to CD path numbers
    rec_user_mflow_path_2 = (flow_path2_data * mult).tolist()
    return rec_user_mflow_path_1, rec_user_mflow_path_2

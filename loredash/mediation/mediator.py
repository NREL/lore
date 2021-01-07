from django.conf import settings
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import time, copy, datetime, math
from twisted.internet.task import LoopingCall
from twisted.internet import reactor
import PySAM_DAOTk.TcsmoltenSalt as pysam
from pathlib import Path
from mediation import data_validator, pysam_wrap, models
import pandas as pd
# import models

class Mediator:
    pysam_wrap = None
    validated_outputs_prev = None
    plant_config = {'design': None, 'location': None }
    default_pysam_model = "MSPTSingleOwner"

    def __init__(self, plant_config={'design': None, 'location': None}, override_with_weather_file_location=True,
                 weather_file=None, preprocess_pysam=True, preprocess_pysam_on_init=True,
                 update_interval=datetime.timedelta(seconds=5), simulation_timestep=datetime.timedelta(minutes=5)):
        self.plant_config = plant_config
        self.override_with_weather_file_location = override_with_weather_file_location
        self.weather_file = weather_file
        self.preprocess_pysam = preprocess_pysam
        self.preprocess_pysam_on_init = preprocess_pysam_on_init
        self.update_interval = update_interval
        self.simulation_timestep = simulation_timestep

        if weather_file is not None and override_with_weather_file_location == True:
            self.plant_config['location'] = GetLocationFromWeatherFile(weather_file)

        self.pysam_wrap = pysam_wrap.PysamWrap(plant_config=self.plant_config, model_name=self.default_pysam_model,
                                               load_defaults=True, weather_file=None,
                                               enable_preprocessing=self.preprocess_pysam,
                                               preprocess_on_init=self.preprocess_pysam_on_init)
    
    def RunOnce(self, datetime_start=None, datetime_end=None):
        """For the current point in time, get data from external plant and weather interfaces and run
        entire set of submodels, saving data to database"""

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
        plant_state = self.GetPlantState()
        
        # b. Call PySAM using inputs
        tech_outputs = self.pysam_wrap.Simulate(datetime_start, datetime_end, self.simulation_timestep, plant_state, weather_dataframe=weather_dataframe)
        print("Annual Energy [kWh]= ", tech_outputs["annual_energy"])

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
        # self.BulkAddToPysamTable(validated_outputs)

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
        datetime_now = datetime.datetime.now()
        datetime_now_rounded_down = RoundMinutes(datetime_now, 'down', self.simulation_timestep.seconds/60)    # the start of the time interval currently in
        datetime_start_prev_day = datetime_now_rounded_down - datetime.timedelta(days=1)
        datetime_end_prev_day = datetime_now_rounded_down                   # end of the last timestep
                                                                            # (as noted for "time_stop" on line 1004 in cmod_tcsmolten_salt.cpp)
        self.RunOnce(datetime_start_prev_day, datetime_end_prev_day)
        return 0

    def BulkAddToPysamTable(self, records):
        n_records = len(records['time_hr'])
        newyears = datetime.datetime(records['year_start'], 1, 1, 0, 0, 0)

        instances = [
            models.PysamData(
                timestamp =             newyears + datetime.timedelta(hours=records['time_hr'][i]),
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

        models.PysamData.objects.bulk_create(instances)

    def GetWeatherDataframe(self, datetime_start, datetime_end, **kwargs):
        """put the weather forecast call here instead"""
        tmy3_path = kwargs.get('tmy3_path') if 'tmy3_path' in kwargs else None
        return Tmy3ToDataframe(tmy3_path, datetime_start, datetime_end)

    def GetPlantState(self):
        """put virtual/real call here instead"""
        plant_state = self.pysam_wrap.GetSimulatedPlantState(self.validated_outputs_prev)      # for initializing next simulation from a prior one
        if plant_state is None:
            plant_state = self.pysam_wrap.GetDefaultPlantState()
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

def RoundMinutes(dt, direction, minute_resolution):
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
        'Latitude': float(df_meta['Latitude'][0]),
        'Longitude': float(df_meta['Longitude'][0]),
        'Time Zone': int(df_meta['Time Zone'][0]),
        'Elevation': float(df_meta['Elevation'][0])
    }
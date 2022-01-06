from math import ceil
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from django.conf import settings
from pathlib import Path
import datetime
import pandas as pd
import json

from mediation import data_validator, ssc_wrap

class TechWrap:
    """A wrapper providing higher-level function calls to the DAOTk power tower
    (custom tcsmolten_salt) model in SSC
    """
    parent_dir = str(Path(__file__).parents[1])
    design_path = parent_dir+"/data/field_design_debugging_only.json"
    MIN_ONE_HOUR_SIMS = True        # used to circumvent SSC bug
    REUSE_FLUXMAPS = True

    def __init__(self, params, plant, dispatch_wrap_params, weather_file=None, start_date_year=2018):
        self.ssc = ssc_wrap.ssc_wrap(wrapper='pyssc', tech_name='tcsmolten_salt', financial_name=None, defaults_name='MSPTSingleOwner') # defaults_name not used for pyssc
        # NOTE: order is important as earlier set parameters are overwritten:
        self.ssc.set(dispatch_wrap_params)
        if not 'solar_resource_data' in plant.design:       # solar_resource_data is where location info is set in ssc
            plant.design['solar_resource_data'] = TechWrap.create_solar_resource_data_var(plant.design)
        self.ssc.set(params)            # what are these exactly?
        self.ssc.set(plant.design)      # Updating from plant.design last to overwrite any plant design defaults in params
        self.set_weather_data(tmy_file_path=weather_file)

        # if (__name__ == "__main__" or settings.DEBUG) == True:
        #     self._set_design_from_file(self.design_path)            # only used for debugging to avoid recalculating flux maps

    def calc_flux_eta_maps(self, plant_design, plant_state):
        """Compute the flux and eta maps and assign them to the ssc model parameters, first getting the needed solar resource data"""
        datetime_start = datetime.datetime(2018, 1, 1, 0, 0, 0)
        solar_resource_data = TechWrap.create_solar_resource_data_var(plant_location=plant_design,     # get 8760 data points to satisfy ssc constraint
                                                                       datetime_start=datetime_start,
                                                                       datetime_end=datetime_start.replace(year=datetime_start.year + 1),
                                                                       timedelta=datetime.timedelta(hours=1))
        datetime_end = datetime_start                                           # run for just first timestep of year
        self.ssc.set({'field_model_type': 2})                                   # generate flux and eta maps but don't optimize field or tower
        original_values = {k:self.ssc.get(k) for k in ['is_dispatch_targets', 'rec_clearsky_model', 'time_steps_per_hour', 'sf_adjust:hourly', ]}
        self.ssc.set({'is_dispatch_targets':False, 'rec_clearsky_model': 1, 'time_steps_per_hour': 1, 'sf_adjust:hourly': [0.0 for j in range(8760)]})    # set so unneeded dispatch targets and clearsky DNI are not required
        self.ssc.set({'sf_adjust:hourly': [0.0 for j in range(8760)]})
        
        tech_outputs = self.simulate(datetime_start, datetime_end, None, plant_state=plant_state, solar_resource_data=solar_resource_data)
        self.ssc.set(original_values)                                           # revert back to original specifications
        eta_map = tech_outputs["eta_map_out"]                                   # get maps and set for subsequent runs
        flux_maps = [r[2:] for r in tech_outputs['flux_maps_for_import']]       # don't include first two columns
        A_sf_in = tech_outputs["A_sf"]
        flux_eta_maps = {'eta_map': eta_map, 'flux_maps': flux_maps, 'A_sf_in': A_sf_in}
        self.ssc.set(flux_eta_maps)
        
        # if __name__ == "__main__" or settings.DEBUG is True:
        #     self._save_design_to_file()

        return flux_eta_maps

    def simulate(self, datetime_start, datetime_end, timestep, plant_state, weather_dataframe=None, solar_resource_data=None):
        """
        datetime_start = beginning of first timestep
        datetime_end = end of last timestep
            (if there is minimum 1-hr simulation duration and datetime_end is less than one hour after datetime_start,
            datetime_end will be set to exactly 1 hour after datetime_start:
            e.g., datetime_end = 18:40 if datetime_start = 17:40)

        returns tech_outputs, where:
            tech_outputs.time_hr = end of each timestep, given as hours since start of current year
        """

        #NOTE: field_model_type values: 0=optimize field and tower; 1=optimize just field based on tower;
        #                               2=no field nor tower optimization; 3=use provided flux and eta maps (don't calculate)
        if not self._design_is_set() or self.REUSE_FLUXMAPS == False:
            self.ssc.set({'field_model_type': 2})           # calculate flux and eta maps at simulation start
        else:
            self.ssc.set({'field_model_type': 3})           # use the provided flux and eta map inputs
            self.ssc.set({'eta_map_aod_format': False})     # false = eta map not in 3D AOD format

        self.ssc.set(plant_state)

        #NOTE: this also pads the weather data if its length is less than required:
        self.set_weather_data(weather_dataframe=weather_dataframe, solar_resource_data=solar_resource_data)

        # set times:
        if self.MIN_ONE_HOUR_SIMS == True:
            datetime_end_original = datetime_end
            datetime_end = max(datetime_end, datetime_start + datetime.timedelta(hours=1))
        datetime_newyears = datetime.datetime(datetime_start.year, 1, 1, 0, 0, 0, tzinfo=datetime_start.tzinfo)
        self.ssc.set({'time_start': (datetime_start - datetime_newyears).total_seconds()})      # time at beginning of first timestep, as
                                                                                                #  seconds since start of current year
        self.ssc.set({'time_stop': (datetime_end - datetime_newyears).total_seconds()})         # time at end of last timestep, as
                                                                                                #  seconds since start of current year
        if timestep is not None:
            self.ssc.set({'time_steps_per_hour': 3600 / timestep.seconds})                      # otherwise its using already set value

        # TODO: do this trimming better and properly ##############################################
        # TODO: array lengths are consistent with ssc requirements, except for sf_adjust:hourly in the flux map call, which is being taken from plant_state
        def resize_list(list_name, total_elements):
            _list = self.ssc.get(list_name)
            if isinstance(_list, int) or isinstance(_list, float):
                _list = total_elements * [_list]
            elif len(_list) < total_elements:
                _list.extend((total_elements - len(_list))*[_list[-1]])
            else:
                _list = _list[0:total_elements]
            self.ssc.set({list_name: _list})

        N_weather_data = len(self.ssc.get('solar_resource_data')['year'])
        resize_list('sf_adjust:hourly', N_weather_data)
        ###########################################################################################

        tech_outputs = self.ssc.execute()
        #print(tech_outputs.keys())

        # Strip trailing zeros or excess data from outputs
        times = {'time_start': tech_outputs['time_start'],
                 'time_stop': tech_outputs['time_stop'],
                 'time_steps_per_hour': tech_outputs['time_steps_per_hour']}
        if self.MIN_ONE_HOUR_SIMS == True:
            times['time_stop'] = (datetime_end_original - datetime_newyears).total_seconds()
        tech_outputs = self._remove_data_padding(tech_outputs, times)

        return tech_outputs

    def set(self, ssc_param_dict):
        self.ssc.set(ssc_param_dict)

    @staticmethod
    def create_solar_resource_data_var(plant_location=None, datetime_start=None, datetime_end=None, timedelta=None):
        """ result: [datetime_start:timedelta:datetime_end) """
        solar_resource_data = {
            'tz':       None,       # [hr]      timezone
            'elev':     None,       # [m]       elevation
            'lat':      None,       # [deg]     latitude
            'lon':      None,       # [deg]     longitude
            'year':     [2018],     # [-]
            'month':    [1],        # [-]
            'day':      [1],        # [-]
            'hour':     [0],        # [hr]
            'minute':   [0],        # [minute]
            'dn':       [0.],       # [W/m2]    DNI
            'df':       [0.],       # [W/m2]    DHI
            'gh':       [0.],       # [W/m2]    GHI
            'wspd':     [0.],       # [m/s]     windspeed
            'tdry':     [0.],       # [C]       ambient dry bulb temperature
            'pres':     [0.],       # [mbar]    ambient pressure
            'tdew':     [0.],       # [C]       dew point
        }

        if plant_location is not None:
            solar_resource_data['tz'] = plant_location['timezone']
            solar_resource_data['elev'] = plant_location['elevation']
            solar_resource_data['lat'] = plant_location['latitude']
            solar_resource_data['lon'] = plant_location['longitude']

        if datetime_start is not None and datetime_end > datetime_start and timedelta is not None:
            date_range = pd.date_range(start=datetime_start,
                                       end=datetime_end - timedelta,
                                       freq=timedelta)
            solar_resource_data['year'] = list(date_range.year)
            solar_resource_data['month'] = list(date_range.month)
            solar_resource_data['day'] = list(date_range.day)
            solar_resource_data['hour'] = list(date_range.hour)
            solar_resource_data['minute'] = list(date_range.minute)
            length = len(date_range)
            solar_resource_data['dn'] = length * [0.]
            solar_resource_data['df'] = length * [0.]
            solar_resource_data['gh'] = length * [0.]
            solar_resource_data['wspd'] = length * [0.]
            solar_resource_data['tdry'] = length * [0.]
            solar_resource_data['pres'] = length * [0.]
            solar_resource_data['tdew'] = length * [0.]

        return solar_resource_data

    @staticmethod
    def weather_df_to_ssc_table(weather_dataframe):
        """solar_resource_data can be directly passed to ssc, after lists are padded to an 8760 length"""
        solar_resource_data = TechWrap.create_solar_resource_data_var()

        solar_resource_data['tz'] = weather_dataframe.attrs['timezone']
        solar_resource_data['elev'] = weather_dataframe.attrs['elevation']
        solar_resource_data['lat'] = weather_dataframe.attrs['latitude']
        solar_resource_data['lon'] = weather_dataframe.attrs['longitude']
        solar_resource_data['year'] = list(weather_dataframe.index.year)
        solar_resource_data['month'] = list(weather_dataframe.index.month)
        solar_resource_data['day'] = list(weather_dataframe.index.day)
        solar_resource_data['hour'] = list(weather_dataframe.index.hour)
        solar_resource_data['minute'] = list(weather_dataframe.index.minute)
        solar_resource_data['dn'] = list(weather_dataframe['DNI'])
        solar_resource_data['df'] = list(weather_dataframe['DHI'])
        solar_resource_data['gh'] = list(weather_dataframe['GHI'])
        solar_resource_data['wspd'] = list(weather_dataframe['Wind Speed'])
        solar_resource_data['tdry'] = list(weather_dataframe['Temperature'])
        solar_resource_data['pres'] = list(weather_dataframe['Pressure'])
        solar_resource_data['tdew'] = list(weather_dataframe['Dew Point'])

        return solar_resource_data

    def set_weather_data(self, tmy_file_path=None, solar_resource_data=None, weather_dataframe=None):
        """
        Set the weather data, using either a TMY file, a solar resource data object, or a weather dataframe.
        Note that solar_resource_data is used by SSC instead of the TMY file if solar_resource_data is assigned
        """

        if isinstance(tmy_file_path, str) and os.path.isfile(tmy_file_path):
            self.ssc.set({'solar_resource_file': tmy_file_path})

        solar_resource_data_input = None
        if weather_dataframe is not None and solar_resource_data is None:
            solar_resource_data_input = TechWrap.weather_df_to_ssc_table(weather_dataframe)     # convert
        elif solar_resource_data is not None:
            solar_resource_data_input = solar_resource_data

        if solar_resource_data_input is not None:
            #TODO: re-enable the following validation. Note that this is also called from
            #      dispatch_wrap via estimates_for_dispatch_model(). From a profiler run,
            #      the following validation, just from estimates_for_dispatch_model(),
            #      took over half the total program execution time.
            validated_solar_resource_data = solar_resource_data_input
            # weather_schema = data_validator.weather_schema      # validate
            # validated_solar_resource_data = weather_schema(solar_resource_data_input)

            # the number of records must be 8760 x ssc 'time_steps_per_hour' input
            # ssc is reads data from the position in the full annual array that corresponds to time_start
            validated_solar_resource_data = self.pad_weather_data(validated_solar_resource_data)
            self.ssc.set({'solar_resource_data': validated_solar_resource_data})
        return 0

    def pad_weather_data(self, solar_resource_data = None, list_data = None, datetime_start = None, timestep = None):
        """
        Pads data to the appropriate length for input to ssc (8760 * time_steps_per_hour), and places non-zero data at the appropriate position in the array 
        If solar_resource_data is supplied, the position in the annual array is determined from the (month, day, hour, minute) contained within solar_resource_data
        If list_data is supplied, then the position in the annual array is determined from the supplied datetime_start and timestep (where datetime_start is fixed-offset time, consistent with a TMY file)
        The list_data input is intended to be used for time-series data not included in the solar_resource_data structure (clearsky DNI, price data, solar field adjustment factors)
        """
        if solar_resource_data is not None:
            n = len(solar_resource_data['dn'])
            datetime_start = datetime.datetime(solar_resource_data['year'][0], solar_resource_data['month'][0], solar_resource_data['day'][0], solar_resource_data['hour'][0], solar_resource_data['minute'][0])
            timestep = datetime.datetime(solar_resource_data['year'][1], solar_resource_data['month'][1], solar_resource_data['day'][1], solar_resource_data['hour'][1], solar_resource_data['minute'][1]) - datetime_start
        elif list_data is not None:  
            n = len(list_data)

        steps_per_hour = int(3600 / timestep.seconds)
        i0 = int((datetime_start-datetime.datetime(datetime_start.year,1,1,0,0,0)).total_seconds() / timestep.seconds) 
        diff = 8760*steps_per_hour - n
        front_padding = [0]*i0
        back_padding = [0]*(diff - i0)

        if solar_resource_data is not None:
            if diff > 0:
                for k in solar_resource_data:
                    if isinstance(solar_resource_data[k],list):
                        solar_resource_data[k] = front_padding + solar_resource_data[k] + back_padding
            return solar_resource_data

        elif list_data is not None:
            if diff > 0:
                padded_data = front_padding + list_data + back_padding
            return padded_data
        
        return 0

    def get_simulated_plant_states(self, model_outputs, **kwargs):
        '''
        Returns the simulated plant states at end of each timestep
        The default assumption is that the trailing zeros from a partial-year simulation have already been stripped
        Inputs:
        model_outputs = outputs dictionary from the technology model, gotten via .Outputs.export()
        strip_zeros = Strip zeroes from a partial-year simulation?
        times = {'time_start' [s], 'time_stop' [s], 'time_steps_per_hour' [1/hr]}
        Returns:
        Dictionary of numbers with the same keys as GetPlantStateIoMap()
        '''

        def get_plant_state_io_map():
            return {
            # Last value in array output becomes number input?
            # Number Inputs                         # Arrays Outputs
            'is_field_tracking_init':               'is_field_tracking_final',
            'rec_op_mode_initial':                  'rec_op_mode_final',
            'rec_startup_time_remain_init':         'rec_startup_time_remain_final',
            'rec_startup_energy_remain_init':       'rec_startup_energy_remain_final',

            'T_tank_cold_init':                     'T_tes_cold',
            'T_tank_hot_init':                      'T_tes_hot',
            'csp_pt_tes_init_hot_htf_percent':      'hot_tank_htf_percent_final',       # in SSC this variable is named csp.pt.tes.init_hot_htf_percent

            'pc_op_mode_initial':                   'pc_op_mode_final',
            'pc_startup_time_remain_init':          'pc_startup_time_remain_final',
            'pc_startup_energy_remain_initial':     'pc_startup_energy_remain_final',

            'wdot0':                                'P_cycle',
            'qdot0':                                'q_pb',
            }

        if model_outputs is None: return None

        if 'strip_zeros' in kwargs and kwargs.get('strip_zeros') == True:
            try:
                self._remove_data_padding(model_outputs, kwargs.get('times'))
            except:
                print("Trailing zeroes could not be stripped. Plant state may be invalid.")

        try:
            plant_state_io_map = get_plant_state_io_map()
            plant_state = {k:model_outputs[v] for (k, v) in plant_state_io_map.items()}
        except:
            plant_state = None
        return plant_state


    def estimates_for_dispatch_model(self, plant_state, datetime_start, horizon, weather_dataframe):

        # Backup parameters (to revert back after simulation)  ->  can I just copy self.tech_model and not have to backup and revert?
        param_names = ['is_dispatch_targets', 'tshours', 'is_rec_startup_trans', 'rec_su_delay', 'rec_qf_delay']
        original_params = {key:self.ssc.get(key) for key in param_names}

        # Set parameters
        datetime_end = datetime_start + datetime.timedelta(hours=horizon)    
        timestep = datetime.timedelta(hours=1/self.ssc.get('time_steps_per_hour'))
        plant_state = plant_state                                              # TODO: plant_design contains all parameters. Could filter.
        self.ssc.set({
            'is_dispatch_targets': False,
            'tshours': 100,                                                     # Inflate TES size so that there is always "somewhere" to put receiver output
            'is_rec_startup_trans': False,
            'rec_su_delay': 0.001,                                              # Simulate with no start-up time to get total available solar energy
            'rec_qf_delay': 0.001,
            })
        weather_dataframe = weather_dataframe
        print(datetime_start, datetime_end, timestep, plant_state, weather_dataframe)
        results = self.simulate(datetime_start, datetime_end, timestep, plant_state=plant_state, weather_dataframe = weather_dataframe)
        
        # Revert back to original parameters
        self.ssc.set(original_params)

        return results


    def _weather_file_is_set(self):
        try:
            solar_resource_file = self.ssc.get('solar_resource_file')   # check if assigned
        except:
            return False

        if isinstance(solar_resource_file, str) and os.path.isfile(solar_resource_file):
            return True
        else:
            return False

    def _design_is_set(self):
        try:
            self.ssc.get('eta_map')      # check if assigned
            self.ssc.get('flux_maps')    # check if assigned
            self.ssc.get('A_sf_in')      # check if assigned
        except:
            return False
        else:
            return True

    def _remove_data_padding(self, model_outputs, times):
        points_per_year = int(times['time_steps_per_hour'] * 24 * 365)
        points_in_simulation = int((times['time_stop']-times['time_start'])/ \
            3600 * times['time_steps_per_hour'])
        import time
        tic = time.process_time()
        for k, v in model_outputs.items():
            if isinstance(v, (list, tuple)) and len(v) == points_per_year:
                model_outputs[k] = v[:points_in_simulation]
        toc = time.process_time()
        # print("Stripping zeroes took {seconds:.2f} seconds".format(seconds=toc-tic))
        return model_outputs

    def _save_design_to_file(self):
        """Saves the applicable plant design parameters, to a json file, from when the flux maps were calculated"""
        try:
            design = {
                'eta_map' : self.ssc.get('eta_map_out'),
                'flux_maps' : [r[2:] for r in self.ssc.get('flux_maps_for_import')],
                'A_sf_in' : self.ssc.get('A_sf'),
                'rec_height' : self.ssc.get('rec_height'),
                'D_rec' : self.ssc.get('D_rec'),
                'h_tower' : self.ssc.get('h_tower'),
            }
            with open(self.design_path, 'w') as design_file:
                json.dump(design, design_file)
        except Exception as err:
            return 1
        else:
            return 0

    def _set_design_from_file(self, file_path):
        """Sets the applicable plant design parameters, from a json file, from when the flux maps were calculated"""
        try:
            with open(file_path, 'r') as design_file:
                design = json.load(design_file)
        except Exception:
            return 1
        else:
            # Verify if these are valid?
            self.ssc.set({'eta_map': design['eta_map']})
            self.ssc.set({'flux_maps': design['flux_maps']})
            self.ssc.set({'A_sf_in': design['A_sf_in']})
            self.ssc.set({'rec_height': design['rec_height']})
            self.ssc.set({'D_rec': design['D_rec']})
            self.ssc.set({'h_tower': design['h_tower']})
            self.ssc.set({'eta_map_aod_format': False})                         # false = eta map not in 3D AOD format
            self.ssc.set({'field_model_type': 3})                               # use the provided flux and eta map inputs
            return 0
    
    # NOTE: not currently used. See dispatch.DispatchTargets, which is used instead. Not sure if this mapping is still useful.
    # @staticmethod
    # def _get_plant_schedules_io_map():
    #     return {
    #     # Array Inputs                          # Array Outputs
    #     'q_pc_target_su_in':                    'q_dot_pc_target_su',
    #     'q_pc_target_on_in':                    'q_dot_pc_target_on',
    #     'q_pc_max_in':                          'q_dot_pc_max',
    #     'is_rec_su_allowed_in':                 'is_rec_su_allowed',
    #     'is_rec_sb_allowed_in':                 '?',    # what is this one?
    #     'is_pc_su_allowed_in':                  'is_pc_su_allowed',
    #     'is_pc_sb_allowed_in':                  'is_pc_sb_allowed',
    #     }

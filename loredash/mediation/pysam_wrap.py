from math import ceil
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from django.conf import settings
from pathlib import Path
import datetime
import os
import json
from mediation import data_validator, mediator, ssc_wrap
import mediation.plant as plant_
import librtdispatch.util as util

class PysamWrap:
    parent_dir = str(Path(__file__).parents[1])
    design_path = parent_dir+"/data/field_design_debugging_only.json"
    MIN_ONE_HOUR_SIMS = True        # used to circumvent SSC bug
    REUSE_FLUXMAPS = True

    def __init__(self, mediator_params, plant, dispatch_wrap_params, weather_file=None, start_date_year=2018):
        self.ssc = ssc_wrap.ssc_wrap(wrapper='pyssc', tech_name='tcsmolten_salt', financial_name=None, defaults_name='MSPTSingleOwner') # defaults_name not used for pyssc
        # NOTE: order is important as earlier set parameters are overwritten:
        self.ssc.set(dispatch_wrap_params)
        if not 'solar_resource_data' in plant.design:       # solar_resource_data is where location info is set in ssc
            plant.design['solar_resource_data'] = PysamWrap.create_solar_resource_data_var(plant.design)
        self.ssc.set(plant.design)
        self.ssc.set(mediator_params)       # what are these exactly?
        self.set_weather_data(tmy_file_path=weather_file)

        self.ssc.set({'sf_adjust:hourly': util.get_field_availability_adjustment(
            mediator_params['time_steps_per_hour'], start_date_year, mediator_params['control_field'],
            mediator_params['use_CD_measured_reflectivity'], plant.design, mediator_params['fixed_soiling_loss'])}
        )
        if all(key in plant.design for key in ('rec_user_mflow_path_1', 'rec_user_mflow_path_1')):
            self.ssc.set({'rec_user_mflow_path_1': plant.design['rec_user_mflow_path_1']})
            self.ssc.set({'rec_user_mflow_path_2': plant.design['rec_user_mflow_path_2']})

        if (__name__ == "__main__" or settings.DEBUG) == True:
            self._set_design_from_file(self.design_path)            # only used for debugging to avoid recalculating flux maps

    def calc_flux_eta_maps(self, weather_dataframe):
        """Compute the flux and eta maps and assign them to the ssc model parameters, first getting the needed solar resource data"""
        self.ssc.set({'field_model_type': 2})                                   # generate flux and eta maps but don't optimize field or tower
        datetime_start = datetime.datetime(2018, 1, 1, 0, 0, 0)
        datetime_end = datetime_start                                           # run for just first timestep of year
        # datetime_end = datetime.datetime(2018, 1, 1, 1, 0, 0)
        # timestep = datetime.timedelta(hours=1)
        tech_outputs = self.Simulate(datetime_start, datetime_end, weather_dataframe=weather_dataframe)
        eta_map = tech_outputs["eta_map_out"]                                   # get maps and set for subsequent runs
        flux_maps = [r[2:] for r in tech_outputs['flux_maps_for_import']]       # Don't include first two columns
        A_sf_in = tech_outputs["A_sf"]
        flux_eta_maps = {'eta_map': eta_map, 'flux_maps': flux_maps, 'A_sf_in': A_sf_in}
        self.ssc.set(flux_eta_maps)

        if __name__ == "__main__" or settings.DEBUG is True:
            self._save_design_to_file()

        return flux_eta_maps

    def Simulate(self, datetime_start, datetime_end, timestep=None, plant_state=None, weather_dataframe=None, solar_resource_data=None):
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
        if not self._DesignIsSet() or self.REUSE_FLUXMAPS == False:
            self.ssc.set({'field_model_type': 2})           # calculate flux and eta maps at simulation start
        else:
            self.ssc.set({'field_model_type': 3})           # use the provided flux and eta map inputs
            self.ssc.set({'eta_map_aod_format': False})     # false = eta map not in 3D AOD format

        if plant_state is None:
            plant_state = plant_.plant_initial_state
        self.ssc.set(plant_state)
        self.set_weather_data(weather_dataframe=weather_dataframe, solar_resource_data=solar_resource_data)

        # set times:
        if self.MIN_ONE_HOUR_SIMS == True:
            datetime_end_original = datetime_end
            datetime_end = max(datetime_end, datetime_start + datetime.timedelta(hours=1))
        datetime_newyears = datetime.datetime(datetime_start.year, 1, 1, 0, 0, 0)
        self.ssc.set({'time_start': (datetime_start - datetime_newyears).total_seconds()})      # time at beginning of first timestep, as
                                                                                                #  seconds since start of current year
        self.ssc.set({'time_stop': (datetime_end - datetime_newyears).total_seconds()})         # time at end of last timestep, as
                                                                                                #  seconds since start of current year
        if timestep is not None:
            self.ssc.set({'time_steps_per_hour': 3600 / timestep.seconds})                      # otherwise its using already set value

        # TODO: do this trimming better and properly ##############################################
        N_weather_vals = len(self.ssc.get('solar_resource_data')['year'])
        self.ssc.set({'sf_adjust:hourly': self.ssc.get('sf_adjust:hourly')[0:N_weather_vals]})
        self.ssc.set({'rec_clearsky_dni': self.ssc.get('rec_clearsky_dni')[0:N_weather_vals]})

        def resize_list(_list, total_elements):
            if len(_list) < total_elements:
                _list.extend((total_elements - len(_list))*[_list[-1]])
                return _list
            else:
                return _list[0:total_elements]
        N_timesteps = ceil((self.ssc.get('time_stop') - self.ssc.get('time_start')) / 3600. * self.ssc.get('time_steps_per_hour'))
        self.ssc.set({'q_pc_target_su_in': resize_list(self.ssc.get('q_pc_target_su_in'), N_timesteps)})
        self.ssc.set({'q_pc_target_on_in': resize_list(self.ssc.get('q_pc_target_on_in'), N_timesteps)})
        self.ssc.set({'q_pc_max_in': resize_list(self.ssc.get('q_pc_max_in'), N_timesteps)})
        self.ssc.set({'is_rec_su_allowed_in': resize_list(self.ssc.get('is_rec_su_allowed_in'), N_timesteps)})
        self.ssc.set({'is_rec_sb_allowed_in': resize_list(self.ssc.get('is_rec_sb_allowed_in'), N_timesteps)})
        self.ssc.set({'is_pc_su_allowed_in': resize_list(self.ssc.get('is_pc_su_allowed_in'), N_timesteps)})
        self.ssc.set({'is_pc_sb_allowed_in': resize_list(self.ssc.get('is_pc_sb_allowed_in'), N_timesteps)})
        # self.ssc.set({'is_ignore_elec_heat_dur_off': resize_list(self.ssc.get('is_ignore_elec_heat_dur_off'), N_timesteps)})    # NOT SET

        ###########################################################################################

        tech_outputs = self.ssc.execute()

        # Strip trailing zeros or excess data from outputs
        times = {'time_start': tech_outputs['time_start'],
                 'time_stop': tech_outputs['time_stop'],
                 'time_steps_per_hour': tech_outputs['time_steps_per_hour']}
        if self.MIN_ONE_HOUR_SIMS == True:
            times['time_stop'] = (datetime_end_original - datetime_newyears).total_seconds()
        tech_outputs = self._RemoveDataPadding(tech_outputs, times)

        return tech_outputs

    @staticmethod
    def create_solar_resource_data_var(plant_location=None):
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
        }

        if plant_location is not None:
            solar_resource_data['tz'] = plant_location['timezone']
            solar_resource_data['elev'] = plant_location['elevation']
            solar_resource_data['lat'] = plant_location['latitude']
            solar_resource_data['lon'] = plant_location['longitude']

        return solar_resource_data

    @staticmethod
    def WeatherDataframeToPysamFormat(weather_dataframe):
        """solar_resource_data can be directly passed to PySAM, after lists are padded to an 8760 length"""
        solar_resource_data = PysamWrap.create_solar_resource_data_var()

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
            solar_resource_data_input = PysamWrap.WeatherDataframeToPysamFormat(weather_dataframe)     # convert
        elif solar_resource_data is not None:
            solar_resource_data_input = solar_resource_data

        if solar_resource_data_input is not None:
            weather_schema = data_validator.weather_schema      # validate
            validated_solar_resource_data = weather_schema(solar_resource_data_input)

            # the number of records must be a integer multiple of 8760
            # see: sam_dev/ssc/ssc/common.cpp, line 1272
            diff = len(validated_solar_resource_data['dn']) % 8760
            if diff > 0:
                padding = [0] * (8760 - diff)
                for v in validated_solar_resource_data.values():
                    if isinstance(v, list):
                        v.extend(padding)
            self.ssc.set({'solar_resource_data': validated_solar_resource_data})
        return 0

    def GetSimulatedPlantState(self, model_outputs, **kwargs):
        '''
        Returns simulated plant state at end of prior simulation, or if no prior simulation, returns None
        The default assumption is that the trailing zeros from a partial-year simulation have already been stripped
        Inputs:
        model_outputs = outputs dictionary from the technology model, gotten via .Outputs.export()
        strip_zeros = Strip zeroes from a partial-year simulation?
        times = {'time_start' [s], 'time_stop' [s], 'time_steps_per_hour' [1/hr]}
        Returns:
        Dictionary of numbers with the same keys as GetPlantStateIoMap()
        '''

        def GetPlantStateIoMap():
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
                self._RemoveDataPadding(model_outputs, kwargs.get('times'))
            except:
                print("Trailing zeroes could not be stripped. Plant state may be invalid.")

        try:
            plant_state_io_map = GetPlantStateIoMap()
            plant_state = {k:model_outputs[v][-1] for (k, v) in plant_state_io_map.items()}      # return last value in each list
        except:
            plant_state = None
        return plant_state


    def estimates_for_dispatch_model(self, plant_design, toy, horizon, weather_data, N_pts_horizon, clearsky_data, start_pt):
        """This is an interchangeable pysam version of ssc_wrapper.estimates_for_dispatch_model()"""

        def time_of_year_to_datetime(year, seconds_since_newyears):
            return datetime.datetime(int(year), 1, 1, 0, 0, 0) + datetime.timedelta(seconds=seconds_since_newyears)

        # Backup parameters (to revert back after simulation)  ->  can I just copy self.tech_model and not have to backup and revert?
        param_names = ['is_dispatch_targets', 'tshours', 'is_rec_startup_trans', 'rec_su_delay', 'rec_qf_delay']
        original_params = {key:self.ssc.get(key) for key in param_names}

        # Set parameters
        datetime_start = time_of_year_to_datetime(weather_data['year'][0], toy)
        datetime_end = datetime_start + datetime.timedelta(seconds=horizon)     # horizon is in seconds
        timestep = datetime.timedelta(hours=1/self.ssc.get('time_steps_per_hour'))
        plant_state = plant_design                                              # TODO: plant_design contains all parameters. Could filter.
        solar_resource_data=weather_data
        self.ssc.set({
            'is_dispatch_targets': False,
            'tshours': 100,                                                     # Inflate TES size so that there is always "somewhere" to put receiver output
            'is_rec_startup_trans': False,
            'rec_su_delay': 0.001,                                              # Simulate with no start-up time to get total available solar energy
            'rec_qf_delay': 0.001,

            'q_pc_target_su_in': [0],
            'q_pc_target_on_in': [0],
            'q_pc_max_in': [0],
            'is_rec_su_allowed_in': [0],
            'is_rec_sb_allowed_in': [0],
            'is_pc_su_allowed_in': [0],
            'is_pc_sb_allowed_in': [0],
            })

        results = self.Simulate(datetime_start, datetime_end, timestep, plant_state=plant_state, solar_resource_data=solar_resource_data)
        
        # Revert back to original parameters
        self.ssc.set(original_params)

        # Filter and adjust results
        retvars = ['Q_thermal', 'm_dot_rec', 'beam', 'clearsky', 'tdry', 'P_tower_pump', 'pparasi']
        if results['clearsky'].max() < 1.e-3:         # Clear-sky data wasn't passed through ssc (ssc controlled from actual DNI, or user-defined flow inputs)
            results['clearsky'] = clearsky_data[start_pt : start_pt + N_pts_horizon]

        return results


    def _WeatherFileIsSet(self):
        try:
            solar_resource_file = self.ssc.get('solar_resource_file')   # check if assigned
        except:
            return False

        if isinstance(solar_resource_file, str) and os.path.isfile(solar_resource_file):
            return True
        else:
            return False

    def _DesignIsSet(self):
        try:
            self.ssc.get('eta_map')      # check if assigned
            self.ssc.get('flux_maps')    # check if assigned
            self.ssc.get('A_sf_in')      # check if assigned
        except:
            return False
        else:
            return True

    def _RemoveDataPadding(self, model_outputs, times):
        points_per_year = int(times['time_steps_per_hour'] * 24 * 365)
        points_in_simulation = int((times['time_stop']-times['time_start'])/ \
            3600 * times['time_steps_per_hour'])
        import time
        tic = time.process_time()
        for k, v in model_outputs.items():
            if isinstance(v, (list, tuple)) and len(v) == points_per_year:
                model_outputs[k] = v[:points_in_simulation]
        toc = time.process_time()
        print("Stripping zeroes took {seconds:.2f} seconds".format(seconds=toc-tic))
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
    # def _GetPlantSchedulesIoMap():
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

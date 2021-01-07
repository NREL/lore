from django.conf import settings
from pathlib import Path
import datetime
import os
import json
import PySAM_DAOTk.TcsmoltenSalt as t
#import PySAM_DAOTk.Grid as g
import PySAM_DAOTk.Singleowner as s
from mediation import data_validator

class PysamWrap:
    parent_dir = str(Path(__file__).parents[1])
    design_path = parent_dir+"/data/field_design.json"

    def __init__(self, plant_config, model_name="MSPTSingleOwner", load_defaults=True, weather_file=None,
                 enable_preprocessing=True, preprocess_on_init=True):
        if load_defaults == True:
            self.tech_model = t.default(model_name)
        else:
            self.tech_model = t.new(model_name)

        self.plant_config = plant_config
        #TODO: load plant_config into tech_model and ensure there is a proper initial return value from GetSimulatedPlantState()

        self.SetWeatherData(tmy_file_path=weather_file)
        self.enable_preprocessing = enable_preprocessing

        if self.enable_preprocessing == True:
            design_not_set = True
            if (__name__ == "__main__" or settings.DEBUG) == True:
                design_not_set = self._SetDesign(self.design_path)   # _SetDesign() only used for debugging
            if design_not_set and preprocess_on_init:
                self.PreProcess()                                   # do this now so no simulation delay later

    def PreProcess(self):
        if not self._WeatherFileIsSet():
            solar_resource_data = PysamWrap.GetSolarResourceDataTemplate(plant_location=self.plant_config.location)
        else:
            solar_resource_data = None

        self.tech_model.HeliostatField.field_model_type = 2                             # generate flux maps
        datetime_start = datetime.datetime(2018, 1, 1, 0, 0, 0)         
        datetime_end = datetime_start                                                   # run for just first hour of year
        timestep = datetime.timedelta(hours=1)
        tech_outputs = self.Simulate(datetime_start, datetime_end, timestep, solar_resource_data=solar_resource_data)
        self.tech_model.HeliostatField.eta_map = tech_outputs["eta_map_out"]            # get maps and set for subsequent runs
        self.tech_model.HeliostatField.flux_maps = [r[2:] for r in tech_outputs['flux_maps_for_import']]    # Don't include first two columns
        self.tech_model.HeliostatField.A_sf_in = tech_outputs["A_sf"]

        if __name__ == "__main__" or settings.DEBUG is True:
            self._SaveDesign()

    def Simulate(self, datetime_start, datetime_end, timestep, plant_state=None, weather_dataframe=None, solar_resource_data=None):
        if not self._DesignIsSet() or self.enable_preprocessing == False:
            self.tech_model.HeliostatField.field_model_type = 2
        else:
            self.tech_model.HeliostatField.field_model_type = 3         # use preprocessed maps
            self.tech_model.HeliostatField.eta_map_aod_format = False

        if plant_state is None:
            plant_state = PysamWrap.GetDefaultPlantState()
        result = self._SetPlantState(plant_state)
        result = self.SetWeatherData(weather_dataframe=weather_dataframe, solar_resource_data=solar_resource_data)

        # set times:
        datetime_newyears = datetime.datetime(datetime_start.year, 1, 1, 0, 0, 0)
        self.tech_model.SystemControl.time_start = (datetime_start - datetime_newyears).total_seconds()
        self.tech_model.SystemControl.time_stop = (datetime_end - datetime_newyears).total_seconds()
        self.tech_model.SystemControl.time_steps_per_hour = 3600 / timestep.seconds

        self.tech_model.execute(1)
        tech_outputs = self.tech_model.Outputs.export()
        # tech_attributes = self.tech_model.export()

        # Strip trailing zeros from outputs
        times = {'time_start': self.tech_model.SystemControl.time_start,
                 'time_stop': self.tech_model.SystemControl.time_stop,
                 'time_steps_per_hour': self.tech_model.SystemControl.time_steps_per_hour}
        tech_outputs = self._StripTrailingZeros(tech_outputs, times)

        return tech_outputs

    @staticmethod
    def GetSolarResourceDataTemplate(plant_location=None):
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
            solar_resource_data['tz'] = plant_location['Time Zone']
            solar_resource_data['elev'] = plant_location['Elevation']
            solar_resource_data['lat'] = plant_location['Latitude']
            solar_resource_data['lon'] = plant_location['Longitude']

        return solar_resource_data

    @staticmethod
    def WeatherDataframeToPysamFormat(weather_dataframe):
        """solar_resource_data can be directly passed to PySAM, after lists are padded to an 8760 length"""
        solar_resource_data = PysamWrap.GetSolarResourceDataTemplate()

        solar_resource_data['tz'] = weather_dataframe.attrs['Time Zone']
        solar_resource_data['elev'] = weather_dataframe.attrs['Elevation']
        solar_resource_data['lat'] = weather_dataframe.attrs['Latitude']
        solar_resource_data['lon'] = weather_dataframe.attrs['Longitude']
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

    @staticmethod
    def GetPlantSchedulesIoMap():
        return {
        # Array Inputs                          # Array Outputs
        'q_pc_max_in':                          'q_dot_pc_max',
        'is_rec_su_allowed_in':                 'is_rec_su_allowed',
        'is_pc_su_allowed_in':                  'is_pc_su_allowed',
        'is_pc_sb_allowed_in':                  'is_pc_sb_allowed',
        'q_pc_target_su_in':                    'q_dot_pc_target_su',
        'q_pc_target_on_in':                    'q_dot_pc_target_on',
        }

    @staticmethod
    def GetPlantStateIoMap():
        return {
        # Last value in array output becomes number input?
        # Number Inputs                         # Arrays Outputs
        'pc_op_mode_initial':                   'pc_op_mode_final',
        'pc_startup_time_remain_init':          'pc_startup_time_remain_final',
        'pc_startup_energy_remain_initial':     'pc_startup_energy_remain_final',
        'is_field_tracking_init':               'is_field_tracking_final',
        'rec_op_mode_initial':                  'rec_op_mode_final',
        'rec_startup_time_remain_init':         'rec_startup_time_remain_final',
        'rec_startup_energy_remain_init':       'rec_startup_energy_remain_final',
        'T_tank_hot_init':                      'T_tes_hot',
        'T_tank_cold_init':                     'T_tes_cold',
        'csp_pt_tes_init_hot_htf_percent':      'hot_tank_htf_percent_final',       # in SSC this variable is named csp.pt.tes.init_hot_htf_percent
        }

    @staticmethod
    def GetDefaultPlantState():
        # NOTE: These are the values of the corresponding outputs after 5 minutes of operation,
        # starting with the PySAM default initialization
        default_plant_state = {
            'pc_op_mode_initial':                   1.,
            'pc_startup_time_remain_init':          0.,
            'pc_startup_energy_remain_initial':     0.,
            'is_field_tracking_init':               0.,
            'rec_op_mode_initial':                  0.,
            'rec_startup_time_remain_init':         0.2,
            'rec_startup_energy_remain_init':       167475728,
            'T_tank_hot_init':                      573.9,
            'T_tank_cold_init':                     290.0,
            'csp_pt_tes_init_hot_htf_percent':      25.0,
        }

        assert set(default_plant_state.keys()) == set(PysamWrap.GetPlantStateIoMap().keys())  # Verify all state values are set
        return default_plant_state

    def SetWeatherData(self, tmy_file_path=None, solar_resource_data=None, weather_dataframe=None):
        """solar_resource_data is used instead of the TMY file if solar_resource_data is assigned"""

        if isinstance(tmy_file_path, str) and os.path.isfile(tmy_file_path):
            self.tech_model.SolarResource.solar_resource_file = tmy_file_path

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
            N_to_pad = 8760 - len(validated_solar_resource_data['month']) % 8760
            padding = [0]*N_to_pad
            {k:(v.extend(padding) if isinstance(v, list) else v) for (k,v) in validated_solar_resource_data.items()}

            self.tech_model.SolarResource.solar_resource_data = validated_solar_resource_data

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
        Dictionary of numbers with the same keys as pysam_wrap::GetPlantStateIoMap()
        '''

        if model_outputs is None: return None

        if 'strip_zeros' in kwargs and kwargs.get('strip_zeros') == True:
            try:
                self._StripTrailingZeros(model_outputs, kwargs.get('times'))
            except:
                print("Trailing zeroes could not be stripped. Plant state may be invalid.")

        try:
            plant_state_io_map = PysamWrap.GetPlantStateIoMap()
            plant_state = {k:model_outputs[v][-1] for (k, v) in plant_state_io_map.items()}      # return last value in each list
        except:
            plant_state = None
        return plant_state

    def _WeatherFileIsSet(self):
        try:
            solar_resource_file = self.tech_model.SolarResource.solar_resource_file   # check if assigned
        except:
            return False

        if isinstance(solar_resource_file, str) and os.path.isfile(solar_resource_file):
            return True
        else:
            return False

    def _DesignIsSet(self):
        try:
            self.tech_model.HeliostatField.eta_map      # check if assigned
            self.tech_model.HeliostatField.flux_maps    # check if assigned
            self.tech_model.HeliostatField.A_sf_in      # check if assigned
        except:
            return False
        else:
            return True

    def _StripTrailingZeros(self, model_outputs, times):
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

    def _SaveDesign(self):
        """helper function"""
        try:
            design = {
                'eta_map' : self.tech_model.Outputs.eta_map_out,
                'flux_maps' : self.tech_model.Outputs.flux_maps_for_import,  
                'A_sf' : self.tech_model.Outputs.A_sf,
                'rec_height' : self.tech_model.TowerAndReceiver.rec_height,
                'D_rec' : self.tech_model.TowerAndReceiver.D_rec,
                'h_tower' : self.tech_model.TowerAndReceiver.h_tower,
            }
            with open(self.design_path, 'w') as design_file:
                json.dump(design, design_file)
        except Exception as err:
            return 1
        else:
            return 0

    def _SetDesign(self, file_path):
        """helper function"""
        try:
            with open(file_path, 'r') as design_file:
                design = json.load(design_file)
        except Exception:
            return 1
        else:
            # Verify if these are valid?
            self.tech_model.HeliostatField.eta_map = design['eta_map']
            self.tech_model.HeliostatField.flux_maps = [r[2:] for r in design['flux_maps']]    # Don't include first two columns
            self.tech_model.HeliostatField.A_sf_in = design['A_sf']
            self.tech_model.TowerAndReceiver.rec_height = design['rec_height']
            self.tech_model.TowerAndReceiver.D_rec = design['D_rec']
            self.tech_model.TowerAndReceiver.h_tower = design['h_tower']
            self.tech_model.HeliostatField.eta_map_aod_format = False
            self.tech_model.HeliostatField.field_model_type = 3      # using user-defined flux and efficiency parameters
            return 0

    def _SetPlantState(self, plant_state):
        try:
            for key in plant_state:
                self.tech_model.value(key, plant_state[key])
        except:
            return 1
        else:
            return 0
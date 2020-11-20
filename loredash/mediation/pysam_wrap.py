from django.conf import settings
from pathlib import Path
import datetime
import json
import PySAM_DAOTk.TcsmoltenSalt as t
#import PySAM_DAOTk.Grid as g
import PySAM_DAOTk.Singleowner as s

class PysamWrap:
    parent_dir = str(Path(__file__).parents[1])
    design_path = parent_dir+"/data/field_design.json"
    model_name = "MSPTSingleOwner"
    weather_file = parent_dir+"/data/daggett_ca_34.865371_-116.783023_psmv3_60_tmy.csv"
    tech_model = None
    enable_preprocessing = True

    def __init__(self):
        self.tech_model = t.default(self.model_name)
        self.tech_model.SolarResource.solar_resource_file = self.weather_file

        if self.enable_preprocessing and (__name__ == "__main__" or settings.DEBUG is True):
            self.SetDesign(self.design_path)

    def Simulate(self, datetime_start, datetime_end, timestep, **kwargs):
        """
        datetime_start (datetime), datetime_end (datetime), timestep (timedelta)
        kwargs (pysam_state)
        """
        if self.enable_preprocessing:
            if not self.DesignIsSet():
            # try:
            #     self.tech_model.HeliostatField.eta_map      # check if assigned
            #     self.tech_model.HeliostatField.flux_maps    # check if assigned
            # except:
                self.PreProcess()                           # calculate and assign the above

            self.tech_model.HeliostatField.field_model_type = 3              # use preprocessed maps
            self.tech_model.HeliostatField.eta_map_aod_format = False
        else:
            self.tech_model.HeliostatField.field_model_type = 2

        tech_outputs = self.SimulatePartialYear(datetime_start, datetime_end, timestep)
        return tech_outputs

    def PreProcess(self):
        self.tech_model.HeliostatField.field_model_type = 2                             # generate flux maps
        datetime_start = datetime.datetime(2021, 1, 1, 0, 0, 0)         
        datetime_end = datetime_start + datetime.timedelta(hours=1)                     # run for just first hour of year
        timestep = datetime.timedelta(hours=1)
        tech_outputs = self.SimulatePartialYear(datetime_start, datetime_end, timestep)
        self.tech_model.HeliostatField.eta_map = tech_outputs["eta_map_out"]            # get maps and set for subsequent runs
        self.tech_model.HeliostatField.flux_maps = [r[2:] for r in tech_outputs['flux_maps_for_import']]    # Don't include first two columns
        self.tech_model.HeliostatField.A_sf_in = tech_outputs["A_sf"]

        if __name__ == "__main__" or settings.DEBUG is True:
            self.SaveDesign()

    def DesignIsSet(self):
        try:
            self.tech_model.HeliostatField.eta_map      # check if assigned
            self.tech_model.HeliostatField.flux_maps    # check if assigned
            self.tech_model.HeliostatField.A_sf_in      # check if assigned
        except:
            return False
        else:
            return True

    def SimulatePartialYear(self, datetime_start, datetime_end, timestep, **kwargs):
        """helper function"""
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
        tech_outputs = self.StripTrailingZeros(tech_outputs, times)

        return tech_outputs

    def StripTrailingZeros(self, model_outputs, times):
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

    def SaveDesign(self):
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

    def SetDesign(self, file_path):
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

    PlantSchedulesIoMap = {
        # Array Inputs                          # Array Outputs
        'q_pc_max_in':                          'q_dot_pc_max',
        'is_rec_su_allowed_in':                 'is_rec_su_allowed',
        'is_pc_su_allowed_in':                  'is_pc_su_allowed',
        'is_pc_sb_allowed_in':                  'is_pc_sb_allowed',
        'q_pc_target_su_in':                    'q_dot_pc_target_su',
        'q_pc_target_on_in':                    'q_dot_pc_target_on',
    }

    PlantStateIoMap = {
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

    def GetDefaultPlantState(self):
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

        assert set(default_plant_state.keys()) == set(self.PlantStateIoMap.keys())  # Verify all state values are set
        return default_plant_state

    def GetSimulatedPlantState(self, model_outputs, **kwargs):
        '''
        Returns simulated plant state at end of prior simulation, or if no prior simulation, returns None
        The default assumption is that the trailing zeros from a partial-year simulation have already been stripped
        Inputs:
        model_outputs = outputs dictionary from the technology model, gotten via .Outputs.export()
        strip_zeros = Strip zeroes from a partial-year simulation?
        times = {'time_start' [s], 'time_stop' [s], 'time_steps_per_hour' [1/hr]}
        Returns:
        Dictionary of numbers with the same keys as pysam_wrap::PlantStateIoMap
        '''

        if model_outputs is None: return None

        if 'strip_zeros' in kwargs and kwargs.get('strip_zeros') == True:
            try:
                self.StripTrailingZeros(model_outputs, kwargs.get('times'))
            except:
                print("Trailing zeroes could not be stripped. Plant state may be invalid.")

        try:
            plant_state = {k:model_outputs[v][-1] for (k, v) in self.PlantStateIoMap.items()}      # return last value in each list
        except:
            plant_state = None
        return plant_state

    def SetPlantState(self, plant_state):
        try:
            for key in plant_state:
                self.tech_model.value(key, plant_state[key])
        except:
            return 0
        else:
            return 1



if __name__ == "__main__":
    """Code for testing:"""

    # Inputs
    timestep = datetime.timedelta(minutes=5)
    datetime_start = datetime.datetime(2021, 1, 1, 0, 0, 0)     # 2020 is a leap year
    # datetime_end = datetime.datetime(2020, 1, 1, 12, 0, 0)
    datetime_end = datetime.datetime(2022, 1, 1, 0, 0, 0)       # end of year

    pysam_wrap = PysamWrap()
    
    # With no preprocessing of design info (for testing):
    pysam_wrap.enable_preprocessing = False
    tech_outputs = pysam_wrap.Simulate(datetime_start, datetime_end, timestep)
    annual_energy_kWh = tech_outputs["annual_energy"]       # full year = 563988036

    # With preprocessing of design info:
    pysam_wrap.enable_preprocessing = True
    tech_outputs = pysam_wrap.Simulate(datetime_start, datetime_end, timestep)
    annual_energy_kWh_2 = tech_outputs["annual_energy"]       # full year = 563988036

    pass

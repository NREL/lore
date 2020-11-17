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

        # remove trailing zeros
        points_per_year = int(self.tech_model.SystemControl.time_steps_per_hour * 24 * 365)
        points_in_simulation = int((self.tech_model.SystemControl.time_stop-self.tech_model.SystemControl.time_start)/ \
            3600 * self.tech_model.SystemControl.time_steps_per_hour)
        
        import time
        tic = time.process_time()
        for k, v in tech_outputs.items():
            if isinstance(v, (list, tuple)) and len(v) == points_per_year:
                tech_outputs[k] = v[:points_in_simulation]
        toc = time.process_time()
        print("Stripping zeroes took {seconds:.2f} seconds".format(seconds=toc-tic))

        return tech_outputs

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

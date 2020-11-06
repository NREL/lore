from django.conf import settings
from pathlib import Path
import datetime
import csv
import PySAM_DAOTk.TcsmoltenSalt as t
#import PySAM_DAOTk.Grid as g
import PySAM_DAOTk.Singleowner as s

class PysamWrap:
    parent_dir = str(Path(__file__).parents[1])
    eta_map_path = parent_dir+"/data/eta_map.csv"
    flux_maps_path = parent_dir+"/data/flux_maps.csv"
    model_name = "MSPTSingleOwner"
    weather_file = parent_dir+"/data/daggett_ca_34.865371_-116.783023_psmv3_60_tmy.csv"
    tech_model = None
    enable_preprocessing = True

    def __init__(self):
        self.tech_model = t.default(self.model_name)
        self.tech_model.SolarResource.solar_resource_file = self.weather_file

        if self.enable_preprocessing and (__name__ == "__main__" or settings.DEBUG is True):
            maps = self.ReadMaps()
            if all(maps.values()):      # if all values not None
                self.tech_model.HeliostatField.eta_map = maps["eta_map"]
                self.tech_model.HeliostatField.flux_maps = maps["flux_maps"]

    def Simulate(self, datetime_start, datetime_end, timestep, **kwargs):
        """
        datetime_start (datetime), datetime_end (datetime), timestep (timedelta)
        kwargs (pysam_state)
        """
        if self.enable_preprocessing:
            try:
                self.tech_model.HeliostatField.eta_map      # check if assigned
                self.tech_model.HeliostatField.flux_maps    # check if assigned
            except:
                self.PreProcess()                           # calculate and assign the above

            self.tech_model.HeliostatField.field_model_type = 3              # use preprocessed maps
            self.tech_model.HeliostatField.eta_map_aod_format = 0
            self.tech_model.HeliostatField.A_sf_in = 1269054.4919999996      # just setting here manually for now
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
        self.tech_model.HeliostatField.flux_maps = tech_outputs["flux_maps_for_import"]

        if __name__ == "__main__" or settings.DEBUG is True:
            self.SaveMaps(tech_outputs["eta_map_out"], tech_outputs["flux_maps_for_import"])

    def SimulatePartialYear(self, datetime_start, datetime_end, timestep, **kwargs):
        """helper function"""
        datetime_newyears = datetime.datetime(datetime_start.year, 1, 1, 0, 0, 0)
        self.tech_model.SystemControl.time_start = (datetime_start - datetime_newyears).total_seconds()
        self.tech_model.SystemControl.time_stop = (datetime_end - datetime_newyears).total_seconds()
        self.tech_model.SystemControl.time_steps_per_hour = 3600 / timestep.seconds

        self.tech_model.execute(1)
        tech_outputs = self.tech_model.Outputs.export()
        # tech_attributes = self.tech_model.export()
        return tech_outputs

    def SaveMaps(self, eta_map, flux_maps):
        """helper function"""
        try:
            with open(self.eta_map_path, 'w', newline='') as eta_map_file:
                wr = csv.writer(eta_map_file, quoting=csv.QUOTE_NONE)
                wr.writerows(eta_map)
            with open(self.flux_maps_path, 'w', newline='') as flux_maps_file:
                wr = csv.writer(flux_maps_file, quoting=csv.QUOTE_NONE)
                wr.writerows(flux_maps)
        except Exception as err:
            return 1
        else:
            return 0

    def ReadMaps(self):
        """helper function"""
        try:
            with open(self.eta_map_path, 'r') as eta_map_file:
                csv_reader = csv.reader(eta_map_file, quoting=csv.QUOTE_NONNUMERIC)
                eta_map = tuple(map(tuple, csv_reader))
            with open(self.flux_maps_path, 'r') as flux_maps_file:
                csv_reader = csv.reader(flux_maps_file, quoting=csv.QUOTE_NONNUMERIC)
                flux_maps = tuple(map(tuple, csv_reader))
        except Exception:
            eta_map = flux_maps = None

        return {"eta_map": eta_map, "flux_maps": flux_maps}

if __name__ == "__main__":
    """Code for testing:"""

    # Inputs
    timestep = datetime.timedelta(minutes=5)
    datetime_start = datetime.datetime(2021, 1, 1, 0, 0, 0)     # 2020 is a leap year
    # datetime_end = datetime.datetime(2020, 1, 1, 12, 0, 0)
    datetime_end = datetime.datetime(2022, 1, 1, 0, 0, 0)       # end of year

    pysam_wrap = PysamWrap()
    
    # With no preprocessing of maps (for testing):
    # pysam_wrap.enable_preprocessing = False
    # tech_outputs = pysam_wrap.Simulate(datetime_start, datetime_end, timestep)
    # annual_energy_kWh = tech_outputs["annual_energy"]       # full year = 563988036

    # With preprocessing of maps:
    pysam_wrap.enable_preprocessing = True
    tech_outputs = pysam_wrap.Simulate(datetime_start, datetime_end, timestep)
    annual_energy_kWh = tech_outputs["annual_energy"]       # full year = 27077180

    pass

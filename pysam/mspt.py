from pathlib import Path
import datetime
import csv
import PySAM_DAOTk.TcsmoltenSalt as t
#import PySAM_DAOTk.Grid as g              # new to 2020.2.29, PySAM still based on 2018.11.11
import PySAM_DAOTk.Singleowner as s

parent_dir = str(Path(__file__).parents[1])
weather_file = parent_dir+"/loredash/data/daggett_ca_34.865371_-116.783023_psmv3_60_tmy.csv"
eta_map_path = parent_dir+"/loredash/data/eta_map.csv"
flux_maps_path = parent_dir+"/loredash/data/flux_maps.csv"
model_name = "MSPTSingleOwner"

tech_model = t.default(model_name)
#tech_model.LocationAndResource.solar_resource_file = weather_file
tech_model.SolarResource.solar_resource_file = weather_file             #DAO-TK
try:
    with open(eta_map_path, 'r') as eta_map_file:
        csv_reader = csv.reader(eta_map_file, quoting=csv.QUOTE_NONNUMERIC)
        eta_map = tuple(map(tuple, csv_reader))
    with open(flux_maps_path, 'r') as flux_maps_file:
        csv_reader = csv.reader(flux_maps_file, quoting=csv.QUOTE_NONNUMERIC)
        flux_maps = tuple(map(tuple, csv_reader))
except FileNotFoundError:
    tech_model.HeliostatField.field_model_type = 2           # generate flux maps
except Exception as err:
    print('Error opening eta_map or flux_maps files:', err)
    tech_model.HeliostatField.field_model_type = 2           # generate flux maps
else:
    tech_model.HeliostatField.eta_map = eta_map
    tech_model.HeliostatField.flux_maps = flux_maps
    tech_model.HeliostatField.field_model_type = 3
    tech_model.HeliostatField.eta_map_aod_format = 0
    tech_model.HeliostatField.A_sf_in = 1269054.4919999996

tech_attributes = tech_model.export()

def SaveMaps(eta_map, flux_maps):
    # If imported, will they be output??
    try:
        with open(eta_map_path, 'w', newline='') as eta_map_file:
            wr = csv.writer(eta_map_file, quoting=csv.QUOTE_NONE)
            wr.writerows(eta_map)
        with open(flux_maps_path, 'w', newline='') as flux_maps_file:
            wr = csv.writer(flux_maps_file, quoting=csv.QUOTE_NONE)
            wr.writerows(flux_maps)
    except Exception as err:
        print('Error saving eta or flux maps:', err)

def SimulateFullYear():
    tech_model.execute(1)
    tech_outputs = tech_model.Outputs.export()
    annual_energy_kWh = tech_outputs["annual_energy"]
    print("The annual energy is {:.2e} kWh, which should be close to 5.64e+08 kWh".format(annual_energy_kWh))

    #grid_model = g.default(model_name)
    #grid_attributes = grid_model.export()
    #grid_model.SystemOutput.gen = tech_model.Outputs.gen
    #grid_model.execute(1)
    #grid_outputs = grid_model.Outputs.export()

    financial_model = s.default(model_name)
    financial_attributes = financial_model.export()
    financial_model.SystemOutput.gen = tech_model.Outputs.gen       #should this be from the grid model instead?
    financial_model.execute(1)
    financial_outputs = financial_model.Outputs.export()
    ppa_price = financial_outputs["ppa_price"]
    print("The ppa_price is {:.1f} cents/kWh, which should be close to 10.1 cents/kWh".format(ppa_price))

    SaveMaps(tech_outputs["eta_map_out"], tech_outputs["flux_maps_for_import"])

def SimulatePartialYear():
    # Inputs
    timestep = datetime.timedelta(minutes=5)
    datetime_start = datetime.datetime(2021, 1, 1, 0, 0, 0)     # 2020 is a leap year
    # datetime_end = datetime.datetime(2020, 1, 1, 12, 0, 0)
    datetime_end = datetime.datetime(2022, 1, 1, 0, 0, 0)       # end of year
    # datetime_end = datetime_start + timestep

    datetime_newyears = datetime.datetime(datetime_start.year, 1, 1, 0, 0, 0)
    tech_model.SystemControl.time_start = (datetime_start - datetime_newyears).total_seconds()
    tech_model.SystemControl.time_stop = (datetime_end - datetime_newyears).total_seconds()
    tech_model.SystemControl.time_steps_per_hour = 3600 / timestep.seconds

    tech_model.execute(1)
    tech_outputs = tech_model.Outputs.export()
    annual_energy_kWh = tech_outputs["annual_energy"]       # full year = 563988036

    SaveMaps(tech_outputs["eta_map_out"], tech_outputs["flux_maps_for_import"])
    pass

# SimulateFullYear()
SimulatePartialYear()
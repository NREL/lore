import PySAM_DAOTk.TcsmoltenSalt as t
#import PySAM_DAOTk.Grid as g              # new to 2020.2.29, PySAM still based on 2018.11.11
import PySAM_DAOTk.Singleowner as s
weather_file = "../site/data/daggett_ca_34.865371_-116.783023_psmv3_60_tmy.csv"
model_name = "MSPTSingleOwner"

tech_model = t.default(model_name)
#tech_model.LocationAndResource.solar_resource_file = weather_file
tech_model.SolarResource.solar_resource_file = weather_file             #DAO-TK
tech_attributes = tech_model.export()
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

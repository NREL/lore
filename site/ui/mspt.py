import PySAM.TcsmoltenSalt as t
from pathlib import Path
import pdb

def get_pysam_data():
    parent_dir = str(Path(__file__).parents[1])
    weather_file = parent_dir+"\\data\\daggett_ca_34.865371_-116.783023_psmv3_60_tmy.csv"
    model_name = "MSPTSingleOwner"
    tech_model = t.default(model_name)
    tech_model.SolarResource.solar_resource_file = weather_file
    tech_model.execute(1)
    tech_outputs = tech_model.Outputs.export()
    pdb.set_trace()
    return tech_outputs

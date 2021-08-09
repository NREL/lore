default_app_config = 'ui.apps.UiConfig'

import rapidjson
from pathlib import Path
import os

parent_dir = str(Path(__file__).parents[1])
plant_design_path = os.path.join(parent_dir, "./config/plant_design.json")
with open(plant_design_path) as f:
        plant_design = rapidjson.load(f, parse_mode=1)
TIMEZONE_STRING = plant_design['timezone_string']

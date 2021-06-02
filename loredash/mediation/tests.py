import pytest

# Documentation
#
#   https://pytest-django.readthedocs.io/en/latest/tutorial.html
#
# Installation
#
#   python -m pip install pytest-django
#
# To run the tests, cd to `/lore/loredash`, then run:
#
#   python -m pytest
#
# To run a single test, e.g.,:
#   pytest mediation/tests.py::test_weather_data_validation
# To debug, add the following to launch.json:
#     {
#         "name": "Python: Pytest",
#         "type": "python",
#         "request": "launch",
#         "module": "pytest",
#         "cwd": "${workspaceFolder}\\loredash"
#     },

import datetime
from pathlib import Path
import numpy as np
import random
import time

from mediation import data_validator
from mediation import dispatch_wrap
from mediation import mediator
from mediation import plant
from mediation import tech_wrap

def test_plant_config_validation():
    plant_config = {
        "name":  "plant_name",
        "location": {
            "latitude":  38.2,
            "longitude": -117.4,
            "elevation": 1524,
            "timezone": -8
        }
    }
    plant_config_schema = data_validator.plant_config_schema
    validated_plant_config = plant_config_schema(plant_config)
    assert validated_plant_config['name'] == plant_config['name']
    assert validated_plant_config['location'] == plant_config['location']
    plant_config['location']['timezone'] = 200
    with pytest.raises(Exception) as excinfo:
        validated_plant_config = plant_config_schema(plant_config)
    assert "value must be at most" in str(excinfo.value)
    return

def test_roundtime():
    dt = datetime.datetime(2021, 1, 4, 23, 59, 59, 510000)
    second_resolution = 1
    dt_rounded = mediator.round_time(dt, second_resolution)
    assert dt_rounded == datetime.datetime(2021, 1, 5, 0, 0, 0, 0)
    dt = datetime.datetime(2021, 1, 4, 23, 59, 59, 500000)
    second_resolution = 1
    dt_rounded = mediator.round_time(dt, second_resolution)
    assert dt_rounded == datetime.datetime(2021, 1, 5, 0, 0, 0, 0)
    dt = datetime.datetime(2021, 1, 4, 23, 59, 59, 490000)
    second_resolution = 1
    dt_rounded = mediator.round_time(dt, second_resolution)
    assert dt_rounded == datetime.datetime(2021, 1, 4, 23, 59, 59, 0)
    return

def test_round_minutes():
    timestep_minutes = 5
    datetime1 = datetime.datetime(2014, 8, 31, 17, 34, 0)
    datetime1_up = mediator.round_minutes(datetime1, 'up', timestep_minutes)
    datetime1_dn = mediator.round_minutes(datetime1, 'down', timestep_minutes)
    assert datetime1_up == datetime.datetime(2014, 8, 31, 17, 35, 0)
    assert datetime1_dn == datetime.datetime(2014, 8, 31, 17, 30, 0)
    timestep_minutes = 5
    datetime2 = datetime.datetime(2014, 8, 31, 17, 35, 0)
    datetime2_up = mediator.round_minutes(datetime2, 'up', timestep_minutes)
    datetime2_dn = mediator.round_minutes(datetime2, 'down', timestep_minutes)
    assert datetime2_up == datetime.datetime(2014, 8, 31, 17, 35, 0)
    assert datetime2_dn == datetime.datetime(2014, 8, 31, 17, 35, 0)
    timestep_minutes = 5
    datetime3 = datetime.datetime(2014, 8, 31, 23, 55, 1)
    datetime3_up = mediator.round_minutes(datetime3, 'up', timestep_minutes)
    datetime3_dn = mediator.round_minutes(datetime3, 'down', timestep_minutes)
    assert datetime3_up == datetime.datetime(2014, 9, 1, 0, 0, 0)
    assert datetime3_dn == datetime.datetime(2014, 8, 31, 23, 55, 0)
    return

def test_tmy_to_dataframe():
    weather_file_path = 'nonsense'
    with pytest.raises(Exception) as excinfo:
        df = mediator.tmy3_to_df(tmy3_path=weather_file_path)
    assert "file not found" in str(excinfo.value)

    parent_dir = str(Path(__file__).parents[1])
    weather_file_path = parent_dir+"/data/daggett_ca_34.865371_-116.783023_psmv3_60_tmy.csv"
    datetime_start = datetime.datetime(2020, 11, 20, 1, 1, 0)
    datetime_end = datetime.datetime(2021, 5, 1, 1, 1, 0)
    df = mediator.tmy3_to_df(
        tmy3_path=weather_file_path,
        datetime_start=datetime_start,
        datetime_end=datetime_end,
    )
    return

def test_weather_data_validation():
    weather_data = tech_wrap.TechWrap.create_solar_resource_data_var()
    weather_data['tz'] = -8.        # [hr]      timezone
    weather_data['elev']= 561.      # [m]       elevation
    weather_data['lat'] = 34.85     # [deg]     latitude
    weather_data['lon'] = -116.78   # [deg]     longitude
    {k:(v.append(v[-1]) if isinstance(v, list) else v) for k,v in weather_data.items()}  # duplicate last item in list
    weather_schema = data_validator.weather_schema
    validated_data = weather_schema(weather_data)
    assert validated_data['tz'] == weather_data['tz']
    assert validated_data['elev'] == weather_data['elev']
    assert validated_data['lat'] == weather_data['lat']
    assert validated_data['lon'] == weather_data['lon']
    assert validated_data['year'] == weather_data['year']
    assert validated_data['month'] == weather_data['month']
    assert validated_data['day'] == weather_data['day']
    assert validated_data['hour'] == weather_data['hour']
    assert validated_data['minute'] == weather_data['minute']
    assert validated_data['dn'] == weather_data['dn']
    assert validated_data['df'] == weather_data['df']
    assert validated_data['gh'] == weather_data['gh']
    assert validated_data['wspd'] == weather_data['wspd']
    assert validated_data['tdry'] == weather_data['tdry']
    # Floats are fine for integers as long as no decimal
    weather_data['year'] = [2018.0, 2018.0]
    validated_data = weather_schema(weather_data)
    assert validated_data['year'] == weather_data['year'] and \
        isinstance(validated_data['year'][0], int) and \
        isinstance(validated_data['year'][1], int)
    weather_data['year'] = [2018.1, 2018.1]
    with pytest.raises(Exception):
        validated_data = weather_schema(weather_data)
    # Missing values should cause exception
    weather_data = tech_wrap.TechWrap.create_solar_resource_data_var()
    weather_data['tz'] = -8.        # [hr]      timezone
    weather_data['elev']= 561.      # [m]       elevation
    weather_data['lat'] = 34.85     # [deg]     latitude
    weather_data['lon'] = -116.78   # [deg]     longitude
    del weather_data['year']
    with pytest.raises(Exception) as excinfo:
        validated_data = weather_schema(weather_data)
    assert "required key not provided" in str(excinfo.value)
    # Unequal list lengths should cause exception
    weather_data = tech_wrap.TechWrap.create_solar_resource_data_var()
    weather_data['tz'] = -8.        # [hr]      timezone
    weather_data['elev']= 561.      # [m]       elevation
    weather_data['lat'] = 34.85     # [deg]     latitude
    weather_data['lon'] = -116.78   # [deg]     longitude
    weather_data['year'].append(weather_data['year'][-1])
    with pytest.raises(Exception) as excinfo:
        validated_data = weather_schema(weather_data)
    assert "list lengths must match" in str(excinfo.value)
    return

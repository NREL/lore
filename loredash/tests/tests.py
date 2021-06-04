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
import copy
from voluptuous import Schema, Required, Optional, Range, And, Or,\
    DefaultTo, SetTo, Any, Coerce, Maybe, ALLOW_EXTRA, All

from mediation import data_validator
from mediation import dispatch_wrap
from mediation import mediator
from mediation import plant as plant_
from mediation import tech_wrap
from data.mspt_2020_defaults import default_ssc_params

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

#TODO: get this test working:
# def test_ssc_one_hour_minimum():
#     parent_dir = str(Path(__file__).parents[1])
#     weather_file = parent_dir+"/data/daggett_ca_34.865371_-116.783023_psmv3_60_tmy.csv"
#     timestep = datetime.timedelta(minutes=5)
#     datetime_start = datetime.datetime(2021, 1, 1, 0, 0, 0)
#     datetime_end = datetime.datetime(2021, 1, 1, 0, 30, 0)

#     plant = plant_.Plant(
#         design=plant_.plant_design,
#         initial_state=plant_.plant_initial_state)

#     default_ssc_params.update(plant.get_state())                                # combine default and plant params, overwriting the defaults
#     default_ssc_params.update(mediator.mediator_params)                         # combine default and mediator params, overwriting the defaults
#     default_ssc_params['rec_clearsky_dni'] = [0]*8760
#     tech_wrap1 = tech_wrap.TechWrap(
#         params=default_ssc_params,                                          # already a copy so tech_wrap cannot edit
#         plant=copy.deepcopy(plant),                                         # copy so tech_wrap cannot edit
#         dispatch_wrap_params=dispatch_wrap.dispatch_wrap_params.copy(),     # copy so tech_wrap cannot edit
#         weather_file=None)

#     tech_outputs = tech_wrap1.simulate(
#         datetime_start,
#         datetime_end,
#         timestep,
#         plant.get_state(),
#         mediator.tmy3_to_df(weather_file))
#     T_tes_cold = tech_outputs["T_tes_cold"]

#     # Ensure values are not zeros and there are the correct number of entries
#     assert np.isclose(len(T_tes_cold), (datetime_end - datetime_start).total_seconds() / timestep.seconds)
#     unique_temps = list(set(T_tes_cold))
#     assert len(unique_temps) > 1 or not np.isclose(unique_temps[0], 0)

#TODO: get this test working:
# def test_ssc_preprocessing_vs_not():
#     parent_dir = str(Path(__file__).parents[1])
#     weather_file = parent_dir+"/data/daggett_ca_34.865371_-116.783023_psmv3_60_tmy.csv"
#     timestep = datetime.timedelta(minutes=5)
#     datetime_start = datetime.datetime(2021, 1, 1, 0, 0, 0)     # 2020 is a leap year
#     datetime_end = datetime.datetime(2021, 12, 31, 23, 0, 0)    # end of year

#     # With no preprocessing of design info (for testing):
#     plant1 = plant_.Plant(
#         design=plant_.plant_design,
#         initial_state=plant_.plant_initial_state)

#     default_ssc_params.update(plant1.get_state())                                # combine default and plant params, overwriting the defaults
#     default_ssc_params.update(mediator.mediator_params)                         # combine default and mediator params, overwriting the defaults
#     default_ssc_params['rec_clearsky_dni'] = [0]*8760
#     tech_wrap1 = tech_wrap.TechWrap(
#         params=default_ssc_params,                                          # already a copy so tech_wrap cannot edit
#         plant=copy.deepcopy(plant1),                                         # copy so tech_wrap cannot edit
#         dispatch_wrap_params=dispatch_wrap.dispatch_wrap_params.copy(),     # copy so tech_wrap cannot edit
#         weather_file=None)

#     tech_outputs1 = tech_wrap1.simulate(
#         datetime_start,
#         datetime_end,
#         timestep,
#         plant1.get_state(),
#         mediator.tmy3_to_df(weather_file))
#     annual_energy_kWh1 = tech_outputs1["annual_energy"]


#     # With preprocessing of design info:
#     plant2 = plant_.Plant(
#         design=plant_.plant_design,
#         initial_state=plant_.plant_initial_state)

#     tech_wrap2 = tech_wrap.TechWrap(
#         params=default_ssc_params,                                          # already a copy so tech_wrap cannot edit
#         plant=copy.deepcopy(plant2),                                         # copy so tech_wrap cannot edit
#         dispatch_wrap_params=dispatch_wrap.dispatch_wrap_params.copy(),     # copy so tech_wrap cannot edit
#         weather_file=None)

#     plant2.update_flux_maps(tech_wrap2.calc_flux_eta_maps(plant2.get_design(), plant2.get_state()))
#     tech_outputs2 = tech_wrap2.simulate(
#         datetime_start,
#         datetime_end,
#         timestep,
#         plant2.get_state(),
#         mediator.tmy3_to_df(weather_file))
#     annual_energy_kWh2 = tech_outputs2["annual_energy"]

#     assert annual_energy_kWh1 == annual_energy_kWh2

def test_voluptuous_functionality():
    kBigNumber = 1.0e10
    kEpsilon = 1.0e-10

    # Normalization (coercion) functions
    def MwToKw(mw):
        try:
            mw = float(mw)
        except:
            return None
        else:
            return mw * 1.e3

    schema = Schema( All(
        # First pass for renaming and normalization
        {
        # Explanation: Since both variables are present and are renamed to the same key,
        #              the latter entry IN TEST_DATA overwrites the first one
        Optional(And('original_name_variation_MW', SetTo('new_name'))): [Coerce(MwToKw)],
        Optional(And('original_name_kW', SetTo('new_name'))): object,           # 'object' means accept value as-is
        },

        # Second pass for validation
        {
        # Explanation:
        # 1. Require the key (e.g., 'missing_name') be present. If not present, set to the one-entry list: [None]
        # 2. Specify the key's value is a list by enclosing conditions in brackets
        # 3. For each list value, try to coerce to a float and then verify it is within a min/max range. If either throws
        #    and exception, set value to None.
        Required('missing_name', default=[None]): [Any(And(Coerce(float), Range(min=0, max=kBigNumber)), SetTo(None))],
        Required('none_value', default=[None]): [Any(And(Coerce(float), Range(min=0, max=kBigNumber)), SetTo(None))],
        Required('new_name', default=[None]): [Any(And(Coerce(float), Range(min=0, max=kBigNumber)), SetTo(None))],
        # You need the 'float' in the following because Range() can't be passed a None value. 'Coerce(float)' from above,
        # and 'float' here both throw an exception if given a None, which is what you want in order to move on to the
        # second Any() parameter, which is 'SetTo(None)'
        Required('unique_name_MW', default=[None]): [Any(And(Coerce(MwToKw), float, \
            Range(min=0, max=kBigNumber)), SetTo(None))],
        Required('lots_of_values_MW', default=[None]): [Any(And(Coerce(MwToKw), float, \
            Range(min=0, max=kBigNumber)), SetTo(None))],
        }
    ),
        extra=ALLOW_EXTRA,          # ALLOW_EXTRA = undefined keys in data won't cause exception
                                    # REMOVE_EXTRA = undefined keys in data will be removed
        required=False              # False = data without defined keys in schema will be kept
    )

    data = {
        'none_value': [None],
        'unique_name_MW': [-2, -1, 5, None],
        'original_name_kW': [-1, 4, -3, None],
        'original_name_variation_MW': [-1, 3, -2, None],
        'lots_of_values_MW': random.sample(range(10, 30000), k=2000)
    }

    tic = time.process_time()
    validated_data = schema(data)
    toc = time.process_time()
    duration_seconds = toc - tic
    assert validated_data['none_value'] == [None]
    assert validated_data['unique_name_MW'] == [None, None, 5000., None]
    assert 'original_name_kW' not in validated_data
    assert 'original_name_variation_MW' not in validated_data
    assert validated_data['new_name'] == [None, 3000., None, None]
    assert validated_data['missing_name'] == [None]
    assert duration_seconds < 0.5

# from django.test import TestCase
import pytest
from voluptuous import Schema, Required, Optional, Range, And, Or, DefaultTo, SetTo, Any, Coerce, Maybe, ALLOW_EXTRA, All
import random, time, copy, datetime
from pathlib import Path
from mediation import pysam_wrap, data_validator, mediator

# To run these tests:
# 1. see: https://pytest-django.readthedocs.io/en/latest/tutorial.html
# 2. cd to /lore/loredash and execute "pytest" in the terminal
# To run a single test, e.g.,:
#   pytest mediation/tests.py::test_weather_data_validation
# To debug, add the following to launch.json:
    # {
    #     "name": "Python: Pytest",
    #     "type": "python",
    #     "request": "launch",
    #     "module": "pytest",
    #     "cwd": "${workspaceFolder}\\loredash"
    # },


# TESTS:

#---PySAM preprocessing vs. not------------------------------------------------------------------------------------------
def test_pysam_preprocessing_vs_not():
    parent_dir = str(Path(__file__).parents[1])
    weather_file = parent_dir+"/data/daggett_ca_34.865371_-116.783023_psmv3_60_tmy.csv"
    plant_config = {'design': None, 'location': None}
    timestep = datetime.timedelta(minutes=5)
    datetime_start = datetime.datetime(2021, 1, 1, 0, 0, 0)     # 2020 is a leap year
    datetime_end = datetime.datetime(2021, 12, 31, 23, 0, 0)    # end of year

    # With no preprocessing of design info (for testing):
    pysam_wrap1 = pysam_wrap.PysamWrap(plant_config=plant_config, load_defaults=True, weather_file=weather_file,
                                       enable_preprocessing=False)
    tech_outputs1 = pysam_wrap1.Simulate(datetime_start, datetime_end, timestep)
    annual_energy_kWh1 = tech_outputs1["annual_energy"]

    # With preprocessing of design info:
    pysam_wrap2 = pysam_wrap.PysamWrap(plant_config=plant_config, load_defaults=True, weather_file=weather_file,
                                       enable_preprocessing=True)
    tech_outputs2 = pysam_wrap2.Simulate(datetime_start, datetime_end, timestep)
    annual_energy_kWh2 = tech_outputs2["annual_energy"]

    assert annual_energy_kWh1 == annual_energy_kWh2
#---/PySAM preprocessing vs. not-----------------------------------------------------------------------------------------

#---PySAM preprocessing--------------------------------------------------------------------------------------------------

# *****************************************************************************************************
# NOTE: THIS TEST PASSED THE LAST TIME IT WAS RUN, BUT IT IS QUITE SLOW SO IT IS NORMALLY COMMENTED OUT
# *****************************************************************************************************

def test_pysam_preprocessing():
    import PySAM_DAOTk.TcsmoltenSalt as t

    #  returns a dictionary of design values
    def run_pysam(weather_file=None, solar_resource_data=None, datetime_start=None, datetime_end=None, timestep=None):
        tech_model = t.default('MSPTSingleOwner')
        if weather_file is not None:
            tech_model.SolarResource.solar_resource_file = weather_file
        if solar_resource_data is not None:
            tech_model.SolarResource.solar_resource_data = solar_resource_data
        if datetime_start is not None and datetime_end is not None and timestep is not None:
            datetime_newyears = datetime.datetime(datetime_start.year, 1, 1, 0, 0, 0)
            tech_model.SystemControl.time_start = (datetime_start - datetime_newyears).total_seconds()
            tech_model.SystemControl.time_stop = (datetime_end - datetime_newyears).total_seconds()
            tech_model.SystemControl.time_steps_per_hour = 3600 / timestep.seconds

        tech_model.HeliostatField.field_model_type = 2                             # generate flux maps

        tech_model.execute(1)

        design = {
        'eta_map' : tech_model.Outputs.eta_map_out,
        'flux_maps' : tech_model.Outputs.flux_maps_for_import,  
        'A_sf' : tech_model.Outputs.A_sf,
        'rec_height' : tech_model.TowerAndReceiver.rec_height,
        'D_rec' : tech_model.TowerAndReceiver.D_rec,
        'h_tower' : tech_model.TowerAndReceiver.h_tower,
        }

        return design

    parent_dir = str(Path(__file__).parents[1])

    #---test 1--------
    weather_file = parent_dir+"/data/daggett_ca_34.865371_-116.783023_psmv3_60_tmy.csv"
    datetime_start = datetime.datetime(2020, 1, 1, 0, 0, 0)
    datetime_end = datetime.datetime(2020, 1, 1, 0, 0, 0)
    timestep = datetime.timedelta(hours=1)
    design_vals1 = run_pysam(weather_file=weather_file, datetime_start=datetime_start, datetime_end=datetime_end, timestep=timestep)

    #---test 2--------
    weather_file = parent_dir+"/data/daggett_ca_34.865371_-116.783023_psmv3_60_tmy.csv"
    design_vals2 = run_pysam(weather_file=weather_file)     # whole year simulation

    assert design_vals2['eta_map'] == design_vals1['eta_map']
    assert design_vals2['flux_maps'] == design_vals1['flux_maps']
    assert design_vals2['A_sf'] == design_vals1['A_sf']
    assert design_vals2['rec_height'] == design_vals1['rec_height']
    assert design_vals2['D_rec'] == design_vals1['D_rec']
    assert design_vals2['h_tower'] == design_vals1['h_tower']

    #---test 3--------
    solar_resource_data = {
        # Corresponds to Daggett, CA
        'tz':       -8,         # [hr]      timezone
        'elev':     561.,       # [m]       elevation
        'lat':      34.85,      # [deg]     latitude
        'lon':      -116.78,    # [deg]     longitude
        'year':     [2018],     # [-]
        'month':    [1],        # [-]
        'day':      [1],        # [-]
        'hour':     [0],        # [hr]
        'minute':   [0],        # [minute]
        'dn':       [0.],       # [W/m2]    DNI
        'df':       [0.],       # [W/m2]    DHI
        'gh':       [0.],       # [W/m2]    GHI
        'wspd':     [3.4],      # [m/s]     windspeed
        'tdry':     [-1.],      # [C]       ambient dry bulb temperature
    }
    # Need to pad data so it is 8760 long
    N_to_pad = 8760 - len(solar_resource_data['month']) % 8760
    padding = [0]*N_to_pad
    {k:(v.extend(padding) if isinstance(v, list) else v) for (k,v) in solar_resource_data.items()}

    datetime_start = datetime.datetime(2018, 1, 1, 0, 0, 0)
    datetime_end = datetime.datetime(2018, 1, 1, 0, 0, 0)
    timestep = datetime.timedelta(hours=1)
    design_vals3 = run_pysam(solar_resource_data=solar_resource_data, datetime_start=datetime_start, datetime_end=datetime_end, timestep=timestep)

    assert design_vals3['eta_map'] == design_vals1['eta_map']
    assert design_vals3['flux_maps'] == design_vals1['flux_maps']
    assert design_vals3['A_sf'] == design_vals1['A_sf']
    assert design_vals3['rec_height'] == design_vals1['rec_height']
    assert design_vals3['D_rec'] == design_vals1['D_rec']
    assert design_vals3['h_tower'] == design_vals1['h_tower']

    #---test 4--------
    solar_resource_data = {
        # Corresponds to Tucson, AZ
        'tz':       -7,         # [hr]      timezone
        'elev':     730.,       # [m]       elevation
        'lat':      32.21,      # [deg]     latitude
        'lon':      -110.98,    # [deg]     longitude
        'year':     [2018],     # [-]
        'month':    [1],        # [-]
        'day':      [1],        # [-]
        'hour':     [0],        # [hr]
        'minute':   [0],        # [minute]
        'dn':       [0.],       # [W/m2]    DNI
        'df':       [0.],       # [W/m2]    DHI
        'gh':       [0.],       # [W/m2]    GHI
        'wspd':     [2.0],      # [m/s]     windspeed
        'tdry':     [7.3],      # [C]       ambient dry bulb temperature
    }
    # Need to pad data so it is 8760 long
    N_to_pad = 8760 - len(solar_resource_data['month']) % 8760
    padding = [0]*N_to_pad
    {k:(v.extend(padding) if isinstance(v, list) else v) for (k,v) in solar_resource_data.items()}

    datetime_start = datetime.datetime(2018, 1, 1, 0, 0, 0)
    datetime_end = datetime.datetime(2018, 1, 1, 0, 0, 0)
    timestep = datetime.timedelta(hours=1)
    design_vals4 = run_pysam(solar_resource_data=solar_resource_data, datetime_start=datetime_start, datetime_end=datetime_end, timestep=timestep)

    assert design_vals4['eta_map'] != design_vals3['eta_map']           # NOT EQUALS!
    assert design_vals4['flux_maps'] != design_vals3['flux_maps']       # NOT EQUALS!
    assert design_vals4['A_sf'] == design_vals3['A_sf']
    assert design_vals4['rec_height'] == design_vals3['rec_height']
    assert design_vals4['D_rec'] == design_vals3['D_rec']
    assert design_vals4['h_tower'] == design_vals3['h_tower']

#---/PySAM preprocessing-------------------------------------------------------------------------------------------------

#---Rounding minutes-----------------------------------------------------------------------------------------------------
def test_round_minutes():
    timestep_minutes = 5
    datetime1 = datetime.datetime(2014, 8, 31, 17, 34, 0)
    datetime1_up = mediator.RoundMinutes(datetime1, 'up', timestep_minutes)
    datetime1_dn = mediator.RoundMinutes(datetime1, 'down', timestep_minutes)
    assert datetime1_up == datetime.datetime(2014, 8, 31, 17, 35, 0)
    assert datetime1_dn == datetime.datetime(2014, 8, 31, 17, 30, 0)

    timestep_minutes = 5
    datetime2 = datetime.datetime(2014, 8, 31, 17, 35, 0)
    datetime2_up = mediator.RoundMinutes(datetime2, 'up', timestep_minutes)
    datetime2_dn = mediator.RoundMinutes(datetime2, 'down', timestep_minutes)
    assert datetime2_up == datetime.datetime(2014, 8, 31, 17, 35, 0)
    assert datetime2_dn == datetime.datetime(2014, 8, 31, 17, 35, 0)

    timestep_minutes = 5
    datetime3 = datetime.datetime(2014, 8, 31, 23, 55, 1)
    datetime3_up = mediator.RoundMinutes(datetime3, 'up', timestep_minutes)
    datetime3_dn = mediator.RoundMinutes(datetime3, 'down', timestep_minutes)
    assert datetime3_up == datetime.datetime(2014, 9, 1, 0, 0, 0)
    assert datetime3_dn == datetime.datetime(2014, 8, 31, 23, 55, 0)
#---/Rounding minutes----------------------------------------------------------------------------------------------------

#---Reading TMY to dataframe---------------------------------------------------------------------------------------------
def test_tmy_to_dataframe():
    weather_file_path = 'nonsense'
    with pytest.raises(Exception) as excinfo:
        df = mediator.Tmy3ToDataframe(tmy3_path=weather_file_path)
    assert "file not found" in str(excinfo.value)

    parent_dir = str(Path(__file__).parents[1])
    weather_file_path = parent_dir+"/data/daggett_ca_34.865371_-116.783023_psmv3_60_tmy.csv"
    datetime_start = datetime.datetime(2020, 11, 20, 1, 1, 0)
    datetime_end = datetime.datetime(2021, 5, 1, 1, 1, 0)
    df = mediator.Tmy3ToDataframe(tmy3_path=weather_file_path, datetime_start=datetime_start, datetime_end=datetime_end)
#---/Reading TMY to dataframe--------------------------------------------------------------------------------------------

#---Solar Resource Data--------------------------------------------------------------------------------------------------
def test_weather_data_validation():
    weather_data = pysam_wrap.PysamWrap.GetSolarResourceDataTemplate()
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
    #
    weather_data['year'] = [2018.1, 2018.1]
    with pytest.raises(Exception):
        validated_data = weather_schema(weather_data)

    # Missing values should cause exception
    weather_data = pysam_wrap.PysamWrap.GetSolarResourceDataTemplate()
    weather_data['tz'] = -8.        # [hr]      timezone
    weather_data['elev']= 561.      # [m]       elevation
    weather_data['lat'] = 34.85     # [deg]     latitude
    weather_data['lon'] = -116.78   # [deg]     longitude
    del weather_data['year']
    with pytest.raises(Exception) as excinfo:
        validated_data = weather_schema(weather_data)
    assert "required key not provided" in str(excinfo.value)

    # Unequal list lengths should cause exception
    weather_data = pysam_wrap.PysamWrap.GetSolarResourceDataTemplate()
    weather_data['tz'] = -8.        # [hr]      timezone
    weather_data['elev']= 561.      # [m]       elevation
    weather_data['lat'] = 34.85     # [deg]     latitude
    weather_data['lon'] = -116.78   # [deg]     longitude
    weather_data['year'].append(weather_data['year'][-1])
    with pytest.raises(Exception) as excinfo:
        validated_data = weather_schema(weather_data)
    assert "list lengths must match" in str(excinfo.value)
#---/Solar Resource Data-------------------------------------------------------------------------------------------------

#---Voluptuous Functionality---------------------------------------------------------------------------------------------
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

#---/Voluptuous Functionality--------------------------------------------------------------------------------------------

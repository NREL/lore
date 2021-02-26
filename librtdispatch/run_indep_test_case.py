# Testing the refactored dispatch functions
from mspt_2020_defaults import vartab as ssc_mspt_default_inputs
import copy
import numpy as np
from case_study import CaseStudy

user_defined_cycle_file = 'udpc_noTamb_dependency.csv'
heliostat_layout_file = './input_files/CD_processed/Crescent_Dunes_heliostat_layout.csv'
ground_truth_weather_file = './model-validation/input_files/weather_files/ssc_weatherfile_1min_2018.csv'  # Weather file derived from CD data
clearsky_file = './model-validation/input_files/weather_files/clearsky_pvlib_ineichen_1min_2018.csv'      # Expected clear-sky DNI from Ineichen model (via pvlib).
price_multiplier_file = 'prices_flat.csv'


def get_plant_design(user_defined_cycle_file, heliostat_layout_file):
    plant_design = plant_design.PlantDesign()           # Default parameters contain best representation of CD plant -> updated in "initialize" depending on specified cycle type

    # Set cycle specifications (from model validation code)  ->  from case_study.py::initialize()
    heliostat_field_file = heliostat_layout_file
    plant_design.P_ref = 120
    plant_design.design_eff = 0.409
    plant_design.T_htf_cold_des = 295.0 # [C]                   # This sets design mass flowrate to that in CD's data
    plant_design.pc_config = 1

    with open(os.path.join(os.path.dirname(__file__), user_defined_cycle_file), 'r') as read_obj:
        csv_reader = reader(read_obj)
        plant_design.ud_ind_od = list(csv_reader)        
    for i in range(len(plant_design.ud_ind_od)):
        plant_design.ud_ind_od[i] = [float(item) for item in plant_design.ud_ind_od[i]]

    plant_design.initialize()                                   # calculate dependent inputs
    return plant_design

def get_plant_properties():
    return plant_design.PlantProperties()                       # Default parameters contain best representation of CD plant and dispatch properties

def get_plant_state(plant_design, plant_properties):
    plant_state = ssc_wrapper.PlantState()                      # TODO: move out of ssc_wrapper
    plant_state.set_default(plant_design, plant_properties)
    return plant_state

def get_historical_weather_data(ground_truth_weather_file, ssc_time_steps_per_hour):
    ground_truth_weather_data = util.read_weather_data(ground_truth_weather_file)
    if ssc_time_steps_per_hour != 60:
        ground_truth_weather_data = util.update_weather_timestep(ground_truth_weather_data, ssc_time_steps_per_hour)
    return ground_truth_weather_data

def get_field_availability_adjustment(ssc_time_steps_per_hour):
    fixed_soiling_loss = 0.02
    adjust = (fixed_soiling_loss * 100 * np.ones(ssc_time_steps_per_hour*24*365))
    adjust = adjust.tolist()
    data_steps_per_hour = len(adjust)/8760  
    if data_steps_per_hour != ssc_time_steps_per_hour:
        adjust = util.translate_to_new_timestep(adjust, 1./data_steps_per_hour, 1./ssc_time_steps_per_hour)
    return adjust

def get_clearsky_data(clearsky_file, ssc_time_steps_per_hour):
    clearsky_data = np.genfromtxt(clearsky_file)
    if ssc_time_steps_per_hour != 60:
        clearsky_data = np.array(util.translate_to_new_timestep(clearsky_data, 1./60, 1./ssc_time_steps_per_hour))
    return clearsky_data

def get_price_data(price_multiplier_file, avg_price, price_steps_per_hour, ssc_time_steps_per_hour):
    price_multipliers = np.genfromtxt(price_multiplier_file)
    if price_steps_per_hour != ssc_time_steps_per_hour:
        price_multipliers = util.translate_to_new_timestep(price_multipliers, 1./price_steps_per_hour, 1./ssc_time_steps_per_hour)
    pmavg = sum(price_multipliers)/len(price_multipliers)  
    price_data = [avg_price*p/pmavg  for p in price_multipliers]  # Electricity price at ssc time steps ($/MWh)
    return price_data


if __name__ == "__main__":

    ssc_time_steps_per_hour = 60               # Simulation time resolution in ssc (1min)
    avg_price = 138
    price_steps_per_hour = 1

    start_time = 24710400
    sim_days = 1

    # Build inputs to dispatch model
    ssc_inputs = copy.deepcopy(ssc_mspt_default_inputs)         # get ssc defaults
    plant_design = get_plant_design(user_defined_cycle_file, heliostat_layout_file)
    ssc_inputs.update(plant_design)                             # add in plant design
    ssc_inputs.update(get_plant_properties())
    ssc_inputs.update(get_plant_flux_maps())
    ssc_inputs.update(get_plant_state(plant_design, get_plant_properties()))

    ground_truth_weather_data = get_historical_weather_data(ground_truth_weather_file, ssc_time_steps_per_hour)
    flux_maps = CaseStudy.simulate_flux_maps(plant_design, get_plant_properties(), ssc_time_steps_per_hour, ground_truth_weather_data)
    ssc_inputs.update(vars(flux_maps))

    ssc_inputs['sf_adjust:hourly'] = get_field_availability_adjustment(ssc_time_steps_per_hour)
    ssc_inputs['rec_clearsky_fraction'] = 1.0
    ssc_inputs['rec_clearsky_model'] = 0
    ssc_inputs['rec_clearsky_dni'] = get_clearsky_data(clearsky_file, ssc_time_steps_per_hour).tolist()
    ssc_inputs['is_dispatch'] = 0                               # Always disable dispatch optimization in ssc
    ssc_inputs['is_dispatch_targets'] = True                    # Because we're using dispatch optimization (not in ssc)
    ssc_inputs['time_start'] = start_time
    ssc_inputs['time_stop'] = start_time + sim_days*24*3600
    ssc_inputs['ppa_multiplier_model'] = 1
    ssc_inputs['dispatch_factors_ts'] = get_price_data(price_multiplier_file, avg_price, price_steps_per_hour, ssc_time_steps_per_hour)
    ssc_inputs['time_steps_per_hour'] = ssc_time_steps_per_hour
    ssc_inputs['is_rec_model_trans'] = False                    # use transient model?
    ssc_inputs['is_rec_startup_trans'] = False                  # use transient startup?
    ssc_inputs['rec_control_per_path'] = True
    ssc_inputs['solar_resource_data'] = ground_truth_weather_data
    ssc_inputs['field_model_type'] = 3
    ssc_inputs['eta_map_aod_format'] = False
    ssc_inputs['is_rec_to_coldtank_allowed'] = True
    ssc_inputs['rec_control_per_path'] = True

    ## Start setting up and calling the model
    

import pandas as pd
import numpy as np
import pickle

import functions
import plotting

from csv import reader
'''
Dates with available 1-min receiver/cycle/TES data: 
    2018_08_31 - 2018_09_06,
    2018_09_08 - 2018_09_09,
    2018_09_13,
    2018_10_07 - 2018_10_31, (note no receiver operation on 2018_10_07, 2018_10_28, and 2018_10_29)
    2019_03_16,
    
Dates with available 1-second receiver data (5 days):
    2018_06_19, 2018_09_12, 2019_01_12, 2019_02_27, 2019_03_24
'''  

all_dates = ['2018_06_19', '2018_08_31', '2018_09_01', '2018_09_02', '2018_09_03', '2018_09_04', '2018_09_05', '2018_09_06', '2018_09_08', '2018_09_09', '2018_09_12', '2018_09_13',
             '2018_10_08', '2018_10_09','2018_10_10','2018_10_11','2018_10_12','2018_10_13','2018_10_14','2018_10_15','2018_10_16','2018_10_17','2018_10_18','2018_10_19','2018_10_20','2018_10_21',
             '2018_10_22','2018_10_23','2018_10_24','2018_10_25','2018_10_26','2018_10_27','2018_10_29','2018_10_30','2018_10_31','2019_01_12','2019_02_27','2019_03_16','2019_03_24']

controller_adj_dates = ['2018_10_09', '2018_10_26', '2018_10_30']           # These dates require a more restrictive method for setting dispatch targets from CD's data

design_params = functions.design_parameters()

use_avg_flow = True # controller issues can arrise when not usign average flow
sliding = True
UD_cycle = True    # Overrides sliding boolean
if UD_cycle:
    design_params.P_ref = 120
    design_params.design_eff = 0.409
    design_params.tshours = 12.0

    design_params.T_htf_cold_des = 295.0 # [C]      # This sets design mass flowrate to that in CD's data
    design_params.cycle_max_frac = 1.2 # [-]        # This allows the model to have greater than design mass flow rates when hot temperature is lower than data for a target heat input

    design_params.pc_config = 1
    #design_params.ud_f_W_dot_cool_des
    #design_params.ud_m_dot_water_cool_des
    fname = 'input_files/udpc_noTamb_dependency.csv'
    #fname = 'input_files/udpc_noTamb_dependency_2MWint_shift.csv'
    with open(fname, 'r') as read_obj:
        csv_reader = reader(read_obj)
        design_params.ud_ind_od = list(csv_reader)

    for i in range(len(design_params.ud_ind_od)):
        design_params.ud_ind_od[i] = [float(item) for item in design_params.ud_ind_od[i]]
elif sliding:
    ### For sliding pressure
    ## These parameters work with heat input calculated using 290 as the lower temperature - however, there are a couple of controller issues
    design_params.P_ref = 125
    design_params.design_eff = 0.378
    design_params.tshours = 10.9
    design_params.tech_type = 3

else:
    ### For fixed pressure
    design_params.P_ref = 120.
    design_params.design_eff = 0.385
    design_params.tshours = 11.5
    design_params.tech_type = 1

    design_params.P_ref = 120
    design_params.design_eff = 0.409
    design_params.tshours = 12.0

props = functions.properties(is_testday=False)   
sim_params = functions.simulation_parameters(simulation_case = 1) 

sim_params.override_receiver_Tout = False
sim_params.time_steps_per_hour = 60  

# Update parameters if needed
props.helio_optical_error_mrad = 2.75  # Heliostat optical error  (combined overall slope, specularity, tracking, canting, etc. error)
props.reflectivity = 'measured' # Heliostat reflectivity: can be a numerical value (0-1), or 'measured' to use average value from CD data on specified day
props.rec_absorptance = 0.94    # Receiver solar absorptivity.  Maximum reasonable value should be 0.96 (when freshly painted)
props.epsilon = 0.88            # Receiver thermal emissivity

# Reading in CD cycle data -> done before simulation to set initial conditions and targets
CD_cycle_dict = functions.read_CD_cycle_data(design_params, use_avg_salt_dens=False, flow_avg_wind=10, dates_w_controller_adj = controller_adj_dates)

# Run single day

date = '2018_10_10' #'2018_09_08' #'2018_10_21'   '2018_10_26'     #'2018_09_09' #'2018_09_06' # #'2018_09_03' #'2018_06_19'
CD_cycle = CD_cycle_dict[date]
props.set_TES_temps(CD_cycle)
if date in controller_adj_dates:
    props.set_dispatch_targets(CD_cycle, design_params, use_avg_flow = use_avg_flow, controller_adj=True)
else:
    props.set_dispatch_targets(CD_cycle, design_params, use_avg_flow = use_avg_flow)
init_tes = functions.estimate_init_TES(CD_cycle, CD_cycle_dict['min_max_tank_levels'])
ssc = functions.run_daily_simulation(date, design_params, props, sim_params, init_TES=init_tes)
CD = functions.read_CD_receiver_data(date, CD_data_direc = 'CD_receiver_data_files/')
## Merging CD data:
for key in CD_cycle.keys():
    CD[key] = np.array(CD_cycle[key])
errors_int, errors_rms = functions.calc_errors(ssc, CD, date, des_params = design_params)   # Calculate full-day integrated errors and RMS errors

### Post-processing
plotting.cycle_validation_plot(ssc, CD_cycle, date, debug=True)


# Multi day run -> could be improved
skip_dates = ['2018_10_07', '2018_10_11', '2018_10_27', '2018_10_28', '2018_10_29']
#skip_dates = []
"""
Reasoning for skip dates:
    2018_10_07 : No power generation and no solar
    2018_10_11 : Poor receiver operations (multiple standby events) 
    2018_10_27 : Very little generation and no solar
    2018_10_28 : No power generation and no solar
    2018_10_29 : No power generation and no solar
"""
first_date = True
plot_days = True
save_plot = True
for date in CD_cycle_dict.keys():  
    if date != 'min_max_tank_levels' and date != 'avg_min_max_tank_levels' and date not in skip_dates:
        if date not in CD_cycle_dict.keys():
            print("No Crescent Dunes dispatch target data for "+ date)
        else:
            CD_cycle = CD_cycle_dict[date]
            print(date)

        props.set_TES_temps(CD_cycle)
        if date in controller_adj_dates:
            props.set_dispatch_targets(CD_cycle, design_params, use_avg_flow = use_avg_flow, controller_adj=True)
        else:
            props.set_dispatch_targets(CD_cycle, design_params, use_avg_flow = use_avg_flow)

        init_tes = functions.estimate_init_TES(CD_cycle, CD_cycle_dict['min_max_tank_levels'])
        ssc = functions.run_daily_simulation(date, design_params, props, sim_params, init_TES=init_tes)
        CD = functions.read_CD_receiver_data(date, CD_data_direc = 'CD_receiver_data_files/')
        ## Merging CD data:
        for key in CD_cycle.keys():
            CD[key] = np.array(CD_cycle[key])
        errors_int, errors_rms = functions.calc_errors(ssc, CD, date, design_params)   # Calculate full-day integrated errors and RMS errors
        if first_date:
            err_int = {key: {} for key in errors_int.keys()}
            err_rms = {key: {} for key in errors_rms.keys()}
            first_date = False
        for key in errors_int.keys():
            err_int[key][date] = errors_int[key]
        for key in errors_rms.keys():
            err_rms[key][date] = errors_rms[key]

        if plot_days:
            plotting.cycle_validation_plot(ssc, CD_cycle, date, save_plot = save_plot)

### find days that meet milestone
MS_dates = []
MS_dates_dtol = []
for day in err_int['gen'].keys():
    if abs(err_int['gen'][day]) < 0.03 and abs(err_int['avg_err_cap'][day]) < 0.05:
        MS_dates.append(day)
    elif abs(err_int['gen'][day]) < 0.03*2 and abs(err_int['avg_err_cap'][day]) < 0.05*2:
        MS_dates_dtol.append(day) 

errors = {}
errors['int'] = err_int
errors['rms'] = err_rms

pickle.dump(errors, open('figures/calc_errors.pkl', "wb") )

plotting.error_plot(err_int['gen'], ylabel = 'Daily Generation Error [%]', save_plot = True, savename = 'gen_error')
plotting.error_plot(err_int['avg_err_cap'], ylabel = 'Normalized Average TES Error [%]', save_plot = True, savename = 'tes_error')


plotting.cum_dist_error_plot(err_int['gen'], milestone = 0.03, xlabel = 'Daily Generation Error [W$_{gross}$ / W$_{gross}$]', save_plot = True, savename = 'gen_error_dist')
plotting.cum_dist_error_plot(err_int['avg_err_cap'], milestone = 0.05, xlabel = r'Normalized Average TES Error [$\bar{e}_{charge}$ / E$_{capacity}$]', save_plot = True, savename = 'tes_error_dist')


### make a plot of errors

# Run multiple days: The same (normalized) annual flux maps are used for each day, so each day must have the same heliostat optical error and attenuation loss
'''
dates = ['2018_09_01', '2018_09_02'] #all_dates
n = len(dates)
errors_int = {k: np.zeros(n) for k in ['Q_thermal', 'mflow', 'Tout']}
errors_rms = {k: np.zeros(n) for k in ['Q_thermal', 'mflow', 'Tout']}
ssc = functions.run_multiple_daily_simulations(dates, design_params, props, sim_params)
for j in range(n):
    CD = functions.read_CD_receiver_data(dates[j], CD_data_direc = 'CD_receiver_data_files/')
    eint, erms = functions.calc_errors(ssc[j], CD, dates[j])
    for k in eint.keys():
        errors_int[k][j] = eint[k]
        errors_rms[k][j] = erms[k]
'''



pass

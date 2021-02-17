
import numpy as np
import functions


'''
Dates with available 1-min receiver/cycle/TES data, heliostat tracking fractions in daily operations report, and per-path mass flow rates: 
    2018_08_31 - 2018_09_06,
    2018_09_08 - 2018_09_09,
    2018_09_13,
    2018_10_07 - 2018_10_31, (note no receiver operation on 2018_10_07 and 2018_10_28)
    2019_03_16,
    
Dates with available 1-second receiver data (5 days), no per-path mass flow rates:
    2018_06_19, 2018_09_12, 2019_01_12, 2019_02_27, 2019_03_24
'''    

all_dates = ['2018_06_19', '2018_08_31', '2018_09_01', '2018_09_02', '2018_09_03', '2018_09_04', '2018_09_05', '2018_09_06', '2018_09_08', '2018_09_09', '2018_09_12', '2018_09_13',
             '2018_10_08', '2018_10_09','2018_10_10','2018_10_11','2018_10_12','2018_10_13','2018_10_14','2018_10_15','2018_10_16','2018_10_17','2018_10_18','2018_10_19','2018_10_20','2018_10_21',
             '2018_10_22','2018_10_23','2018_10_24','2018_10_25','2018_10_26','2018_10_27','2018_10_29','2018_10_30','2018_10_31','2019_01_12','2019_02_27','2019_03_16','2019_03_24']




design_params = functions.design_parameters()   
props = functions.properties()   
sim_params = functions.simulation_parameters(simulation_case = 1) 

sim_params.time_steps_per_hour = 60  

sim_params.use_mflow_per_path = True  # Use separate mass flow rates per receiver flow circuit
sim_params.tracking_fraction_from_op_report = True  # Use tracking fractions taken from the CD daily operations reports instead of the HFCS log files


# Update parameters if needed
props.helio_optical_error_mrad = 2.5  # Heliostat optical error  (combined overall slope, specularity, tracking, canting, etc. error)
props.reflectivity = 'measured' # Heliostat reflectivity: can be a numerical value (0-1), or 'measured' to use average value from CD data on specified day
props.rec_absorptance = 0.94    # Receiver solar absorptivity.  Maximum reasonable value should be 0.96 (when freshly painted)
props.epsilon = 0.88            # Receiver thermal emissivity



# Run single day
date = '2018_10_13' #'2018_06_19'
ssc = functions.run_daily_simulation(date, design_params, props, sim_params)
CD = functions.read_CD_receiver_data(date, CD_data_direc = 'CD_receiver_data_files/')
errors_int, errors_rms = functions.calc_errors(ssc, CD, date, design_params)   # Calculate full-day integrated errors and RMS errors



# Run multiple days: The same (normalized) annual flux maps are used for each day, so each day must have the same heliostat optical error and attenuation loss
'''
dates = all_dates   #['2018_09_01', '2018_09_02']
n = len(dates)
errors_int = {k: np.zeros(n) for k in ['Q_thermal', 'mflow', 'mflow1', 'mflow2', 'Tout', 'Tout1', 'Tout2']}
errors_rms = {k: np.zeros(n) for k in ['Q_thermal', 'mflow', 'mflow1', 'mflow2', 'Tout', 'Tout1', 'Tout2']}
ssc = functions.run_multiple_daily_simulations(dates, design_params, props, sim_params)
for j in range(n):
    CD = functions.read_CD_receiver_data(dates[j], CD_data_direc = 'CD_receiver_data_files/')
    eint, erms = functions.calc_errors(ssc[j], CD, dates[j])
    for k in eint.keys():
        errors_int[k][j] = eint[k]
        errors_rms[k][j] = erms[k]
'''


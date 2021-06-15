from __future__ import print_function
import numpy as np
import pybobyqa
import logging


import functions
import os

'''
Dates with available 1-min receiver/cycle/TES data, heliostat tracking fractions
in daily operations report, and per-path mass flow rates:

    2018_08_31 - 2018_09_06,
    2018_09_08 - 2018_09_09,
    2018_09_13,
    2018_10_07 - 2018_10_31, (excl. 2018_10_07 and 2018_10_28)
    2019_03_16,

Dates with available 1-second receiver data (5 days), no per-path mass flow
rates:

    2018_06_19, 2018_09_12, 2019_01_12, 2019_02_27, 2019_03_24
'''
ALL_DATES = [
    # '2018_06_19',
    '2018_08_31', '2018_09_01', '2018_09_02', '2018_09_03',
    '2018_09_04', '2018_09_05', '2018_09_06', '2018_09_08', '2018_09_09',
    # '2018_09_12',
    '2018_09_13', '2018_10_08', '2018_10_09', '2018_10_10',
    '2018_10_11', '2018_10_12', '2018_10_13', '2018_10_14', '2018_10_15',
    '2018_10_16', '2018_10_17', '2018_10_18', '2018_10_19', '2018_10_20',
    '2018_10_21', '2018_10_22', '2018_10_23', '2018_10_24', '2018_10_25',
    '2018_10_26', '2018_10_27', '2018_10_29', '2018_10_30', '2018_10_31',
    # '2019_01_12',
    # '2019_02_27',
    '2019_03_16',
    # '2019_03_24'
]

def generate_black_box_function(dates = ALL_DATES):
    def black_box_function(
        x, 
        dates = dates,
        output_variables = ['Q_thermal', 'mflow', 'Tout']
    ):
        # Simulation properties
        # Note that we cast all the inputs to `float`, because B.O. uses
        # <numpy.float64> and SSC doesn't like that.

        # Parase the x vector to the correct parameters
        helio_optical_error_mrad = x[0]
        reflectivity = x[1]
        rec_absorptance = x[2]
        hl_ffact = x[3]
        epsilon = x[4]

        props = functions.properties()
        props.helio_optical_error_mrad = float(helio_optical_error_mrad)
        props.reflectivity = float(reflectivity)
        props.rec_absorptance = float(rec_absorptance)
        props.epsilon = float(epsilon)
        props.hl_ffact = float(hl_ffact)

        # Simulation parameters
        sim_params = functions.simulation_parameters(simulation_case = 1)
        sim_params.time_steps_per_hour = 60
        sim_params.override_receiver_Tout = False

        # Design parameters
        design_params = functions.design_parameters()

        # Run SSC
        ssc = functions.run_multiple_daily_simulations(
            dates, design_params, props, sim_params
        )

        # Compute loss
        total_rms_error = 0
        for (i, date) in enumerate(dates):
            CD = functions.read_CD_receiver_data(
                date, CD_data_direc = 'CD_receiver_data_files/'
            )
            int_error, rms = functions.calc_errors(ssc[i], CD, date, design_params, is_avg_tout=True)
            for key in output_variables:
                total_rms_error += rms[key]
            return total_rms_error
    return black_box_function

def main(dates = ALL_DATES, log_filename = "logs.json", max_iter = 100):
    log_filename = "./output_files/per_day/%s" % ("DFO_GM_" + log_filename)

    # Bounded region of parameter space
    pbounds = {
        'helio_optical_error_mrad': (1.5, 4.5),
        'reflectivity': (0.1, 1),
        'rec_absorptance': (0.1, 1),
        'hl_ffact': (0.1, 2),
        'epsilon': (0.1, 1)
    }
    
    # inital point to start
    x0 = np.array([3.0, 0.85, 0.93, 1.0, 0.88])
    # bounds
    upper = np.array([4.5,1,1,2,1])
    lower = np.array([1.5,0.1,0.1,0.1,0.1])
   
    user_params = {'logging.save_diagnostic_info': True, 'logging.save_xk': True}
    soln1 = pybobyqa.solve(generate_black_box_function(dates), x0, bounds=(upper, lower), scaling_within_bounds=True, user_params=user_params, maxfun=max_iter, objfun_has_noise=False, seek_global_minimum=True)
    # print(soln1)
    soln1.diagnostic_info.to_csv(log_filename)
    

import sys
if __name__ == "__main__":
    modulo = int(sys.argv[1])
    offset = int(sys.argv[2])
    for i in range(offset, len(ALL_DATES), modulo):
        main(
            dates = [ALL_DATES[i]],
            log_filename = '%s.csv' % ALL_DATES[i]
        )
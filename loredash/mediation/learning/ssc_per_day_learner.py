# To run this code, you need to set LD_LIBRARY_PATH so it includes the ssc.so
# binary. For example:
#
#   export LD_LIBRARY_PATH=/home/gchen/bo_code_model_validation/dao-tk-project/code/model-validation/; python ssc_learner.py modulo offset
#
# The arguments `modulo` and `offset` control how the process steps through
# ALL_DATES so we can run this script in parallel. To run in serial, use:
#
#   python ssc_learner.py 1 0
#
# To run in parallel, use something like
#
#   python ssc_learner.py 3 0
#   python ssc_learner.py 3 1
#   python ssc_learner.py 3 2
#
# Before running, you need to extract CD_receiver_data_files to a folder of the
# same name.

from bayes_opt import BayesianOptimization
from bayes_opt import event
from bayes_opt import logger
from bayes_opt import util

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

def generate_black_box_function(dates = ALL_DATES, loss_type = "rms"):
    def black_box_function(
        helio_optical_error_mrad = 3.0,
        reflectivity = 0.85,
        rec_absorptance = 0.93,
        hl_ffact = 1.0,
        epsilon = 0.88,
        dates = dates,
        loss_type = loss_type,
        output_variables = ['Q_thermal', 'mflow', 'Tout']
    ):
        # Simulation properties
        # Note that we cast all the inputs to `float`, because B.O. uses
        # <numpy.float64> and SSC doesn't like that.
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
        total_int_error = 0
        for (i, date) in enumerate(dates):
            CD = functions.read_CD_receiver_data(
                date, CD_data_direc = 'CD_receiver_data_files/'
            )
            int_error, rms = functions.calc_errors(ssc[i], CD, date, design_params, is_avg_tout=True)
            for key in output_variables:
                total_rms_error += rms[key]
                total_int_error += abs(int_error[key])
        # Maximizing, so return the negative loss.
        if loss_type == "rms":
            return -total_rms_error
        elif loss_type == "int":
            return -total_int_error
    return black_box_function

def main(dates = ALL_DATES, log_filename = "logs.json", loss_type="rms", max_iter = 100):
    log_filename = "./output_files/per_day/%s" % (loss_type + "_per_day_1_" + log_filename)

    # Bounded region of parameter space
    pbounds = {
        'helio_optical_error_mrad': (1.5, 4.5),
        'reflectivity': (0.1, 1.0),
        'rec_absorptance': (0.1,1),
        'hl_ffact': (0.1, 2),
        'epsilon': (0.1, 1)
    }

    optimizer = BayesianOptimization(
        f = generate_black_box_function(dates, loss_type),
        pbounds = pbounds,
        random_state = 1,
    )

    optimizer.subscribe(
        event.Events.OPTIMIZATION_STEP, logger.JSONLogger(path = log_filename)
    )
    if os.path.isfile(log_filename):
        print("Loading log from %s" % log_filename)
        util.load_logs(optimizer, logs = [log_filename])

    optimizer.maximize(
        init_points = 5,
        n_iter = max_iter,
        acq = 'ei',
        kappa = 2.576,
        kappa_decay = 0.99,
        kappa_decay_delay = 0,
        xi = 0.0
    )

    print(optimizer.max)

import sys
if __name__ == "__main__":
    modulo = int(sys.argv[1])
    offset = int(sys.argv[2])
    # loss = sys.argv[3]
    for i in range(offset, len(ALL_DATES), modulo):
        main(
            dates = [ALL_DATES[i]],
            log_filename = '%s.json' % ("500_" + ALL_DATES[i]),
            loss_type = "rms",
            max_iter = 500
        )

import datetime
import timeit
import numpy as np
import os
import math

import define_cases
import plots
import case_study

os.chdir(os.path.dirname(__file__))

CD_raw_data_direc = './input_files/CD_raw'                      # Directory containing raw data files from CD
CD_processed_data_direc = './input_files/CD_processed'          # Directory containing files with 1min data already extracted

start_date = datetime.datetime(2018, 10, 14)  
sim_days = 1
save_outputs = True
create_plot = True
name = '2019_10_14'

cs = case_study.CaseStudy(isdebug = False)
cs.control_field = 'ssc'
cs.control_receiver = 'ssc_clearsky'
cs.is_optimize = True   
cs.dispatch_weather_horizon = 2  # TODO: what horizon do we want to use?
cs.use_CD_measured_reflectivity = False  
cs.fixed_soiling_loss = 0.02   # 1 - (reflectivity / clean reflectivity)
cs.use_day_ahead_schedule = True
cs.day_ahead_schedule_from = 'calculated'
cs.CD_raw_data_direc = CD_raw_data_direc
cs.CD_processed_data_direc = CD_processed_data_direc
cs.start_date = start_date
cs.sim_days = sim_days
cs.set_initial_state_from_CD_data = True
cs.save_results_to_file = save_outputs           # File name to save results
cs.results_file = name + '_' + '7b'
cs.store_full_dispatch_solns = False  # This keeps a copy of every set inputs/outputs from the dispatch model calls in memory... useful for debugging, but might want to turn off when running large simulations
       
start = timeit.default_timer()
cs.run()
elapsed = timeit.default_timer() - start
print ('Total time elapsed = %.2fs'%(timeit.default_timer() - start))
print ('Receiver thermal generation = %.5f GWht'%cs.total_receiver_thermal)
print ('Cycle gross generation = %.5f GWhe'%cs.total_cycle_gross )
print ('Cycle net generation = %.5f GWhe'%cs.total_cycle_net )
print ('Receiver starts = %d completed, %d attempted'%(cs.n_starts_rec, cs.n_starts_rec_attempted))
print ('Cycle starts = %d completed, %d attempted'%(cs.n_starts_cycle, cs.n_starts_cycle_attempted))
print ('Cycle ramp-up = %.3f'%cs.cycle_ramp_up)
print ('Cycle ramp-down = %.3f'%cs.cycle_ramp_down)
print ('Total under-generation from schedule (beyond tolerance) = %.3f MWhe (ssc), %.3f MWhe (dispatch)'
       %(cs.day_ahead_diff_over_tol_minus['ssc'], cs.day_ahead_diff_over_tol_minus['disp']))
print ('Total over-generation from schedule  (beyond tolerance)  = %.3f MWhe (ssc), %.3f MWhe (dispatch)'
       %(cs.day_ahead_diff_over_tol_plus['ssc'], cs.day_ahead_diff_over_tol_plus['disp']))
print ('Revenue = $%.2f'%cs.revenue)
print ('Day-ahead schedule penalty = $%.2f (ssc), $%.2f (dispatch)'%(cs.day_ahead_penalty_tot['ssc'], cs.day_ahead_penalty_tot['disp']))
print ('Startup/ramping penalty = $%.2f'%cs.startup_ramping_penalty)

# Basic regression tests for refactoring
assert math.isclose(cs.total_receiver_thermal, 3.85, rel_tol=1e-3)
assert math.isclose(cs.total_cycle_gross, 1.89, rel_tol=1e-2)
assert math.isclose(cs.total_cycle_net, 1.74, rel_tol=1e-2)
assert math.isclose(cs.cycle_ramp_up, 121, rel_tol=1e-3)
assert math.isclose(cs.cycle_ramp_down, 121, rel_tol=1e-3)
assert math.isclose(cs.revenue, 243422, rel_tol=1e-2)
assert math.isclose(cs.startup_ramping_penalty, 7200, rel_tol=1e-3)
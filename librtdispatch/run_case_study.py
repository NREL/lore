import datetime
import timeit
import numpy as np
import os
import math

import define_cases
import plots
import case_study

os.chdir(os.path.dirname(__file__))


CD_raw_data_direc = '../../../Crescent Dunes data/NREL - CD collaboration/Steam Generation/Daily Reports/'  # Directory containing raw data files from CD
CD_processed_data_direc = '../../../Crescent Dunes data/Daily operations data/'                             # Directory containing files with 1min data already extracted

start_date = datetime.datetime(2018, 10, 8)
sim_days = 23
save_outputs = True
create_plot = True
name = './outputs/2019_10_08debug'

# cases = ['1a', '2a', '2b', '3a', '4a', '5a', '6a', '7a', '7b']
cases = ['7b']

for c in cases:
    cs = case_study.CaseStudy(isdebug = True)
    define_cases.set_inputs_from_name(cs, c)  # Set control conditions
    
    # Inputs constant for all cases (in addition to those defined by default in CaseStudy)
    cs.CD_raw_data_direc = CD_raw_data_direc
    cs.CD_processed_data_direc = CD_processed_data_direc
    cs.start_date = start_date
    cs.sim_days = sim_days
    cs.set_initial_state_from_CD_data = True

    # File name to save results
    cs.save_results_to_file = save_outputs 
    cs.results_file = name + '_' + c   
    
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

    if create_plot:
        plots.plot_solution(cs, savename = name + '_' + c)   
        #plots.plot_dispatch_soln_states(cs)
        #plots.plot_dispatch_soln_singlevar(cs,'thermal_input_to_cycle')
        #plots.plot_dispatch_soln_singlevar(cs,'receiver_power')
        #plots.plot_dispatch_soln_singlevar(cs,'tes_soc')
    
    
    # Basic regression tests for refactoring
    assert math.isclose(cs.total_receiver_thermal, 3.85, rel_tol=1e-3)
    assert math.isclose(cs.total_cycle_gross, 1.89, rel_tol=1e-2)
    assert math.isclose(cs.total_cycle_net, 1.74, rel_tol=1e-2)
    assert math.isclose(cs.cycle_ramp_up, 121, rel_tol=1e-3)
    assert math.isclose(cs.cycle_ramp_down, 121, rel_tol=1e-3)
    assert math.isclose(cs.revenue, 243422, rel_tol=1e-2)
    assert math.isclose(cs.startup_ramping_penalty, 7200, rel_tol=1e-3)

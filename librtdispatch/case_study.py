import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
from copy import deepcopy
import timeit
import pandas
import os
import math
import datetime

import util
import dispatch
from dispatch import DispatchWrap
import ssc_wrapper
import loredash.mediation.plant as plant_



mediator_params = {
    # Control conditions
	'time_steps_per_hour':			        60,			            # Simulation time resolution in ssc (1min)   DUPLICATED to: ssc_time_steps_per_hour
	'is_dispatch':					        0,                      # Always disable dispatch optimization in ssc
	'is_dispatch_targets':			        True,		            # True if (is_optimize or control_cycle == 'CD_data')
    'is_optimize':					        True,                   # Use dispatch optimization
	'control_field':				        'ssc',                  #'CD_data' = use CD data to control heliostats tracking, heliostats offline, and heliostat daily reflectivity.  Receiver startup time is set to zero so that simulated receiver starts when CD tracking begins
                                                                    #'ssc' = allow ssc to control heliostat field operations, assuming all heliostats are available

	'control_receiver':				        'ssc_clearsky',         #'CD_data' = use CD data to control receiver mass flow. Receiver startup time is set to zero so that simulated receiver starts when CD receiver finishes startup
                                                                    #'ssc_clearsky' = use expected clearsky DNI to control receiver mass flow.  If field_control = 'ssc' then the historical median startup time from CD data will be used to control receiver startup
                                                                    #'ssc_actual_dni' = use actual DNI to control receiver mass flow.  If field_control = 'ssc' then the historical median startup time from CD data will be used to control receiver startup   

	'control_cycle':				        'ssc_heuristic',        # Only used if is_optimize = False
                                                                    # 'CD_data' = use CD actual cycle operation to control cycle dispatch targets
                                                                    # 'ssc_heuristic' = allow ssc heuristic (no consideration of TOD price) to control cycle dispatch

    # Price
    'price_multiplier_file':                'prices_flat.csv',
	'ppa_multiplier_model':			        1,
	'price_steps_per_hour':			        1,                      # Number of steps per hour in electricity price multipliers
	'avg_price':					        138,                    # Average electricity price ($/MWh):  CD original PPA was $138/MWh
    'avg_purchase_price':                   30,                     # Average electricity purchase price ($/MWh) (note, this isn't currently used in the dispatch model)
    'avg_price_disp_storage_incentive':     0.0,                    # Average electricity price ($/MWh) used in dispatch model storage inventory incentive

    # Field, receiver, and cycle simulation options
    'ground_truth_weather_file':            './model-validation/input_files/weather_files/ssc_weatherfile_1min_2018.csv',   # Weather file derived from CD data: DNI, ambient temperature,
                                                                                                                            #  wind speed, etc. are averaged over 4 CD weather stations,
                                                                                                                            #  after filtering DNI readings for bad measurements. 
    'clearsky_file':                        './model-validation/input_files/weather_files/clearsky_pvlib_ineichen_1min_2018.csv',   # Expected clear-sky DNI from Ineichen model (via pvlib). 
    'CD_mflow_path1_file':                  './model-validation/input_files/mflow_path1_2018_1min.csv',  # File containing CD data for receiver path 1 mass flow rate (note, all values are zeros on days without data)
    'CD_mflow_path2_file':                  './model-validation/input_files/mflow_path2_2018_1min.csv',  # File containing CD data for receiver path 2 mass flow rate (note, all values are zeros on days without data)
    'CD_raw_data_direc':                    './input_files/CD_raw',                                      # Directory containing raw data files from CD
    'CD_processed_data_direc':              './input_files/CD_processed',                                # Directory containing files with 1min data already extracted
	'rec_control_per_path':			        True,
	'field_model_type':				        3,
	'eta_map_aod_format':			        False,
	'is_rec_user_mflow':			        False,		            # or this should be unassigned, True if control_receiver == 'CD_data'
	'rec_clearsky_fraction':		        1.0,
    'rec_clearsky_model':                   0,
    'rec_su_delay':                         0.,                     # = 0.01 if control_field == 'CD_data' or control_receiver == 'CD_data'  Set receiver start time and energy to near zero to enforce CD receiver startup timing
    'rec_qf_delay':                         0.,                     # = 0.01 if control_field == 'CD_data' or control_receiver == 'CD_data'
    'is_rec_to_coldtank_allowed':           True,
    'use_CD_measured_reflectivity':	        False,                  # Use measured heliostat reflectivity from CD data
    'fixed_soiling_loss':			        0.02,                   # Fixed soiling loss (if not using CD measured reflectivity) = 1 - (reflectivity / clean_reflectivity)
	'is_rec_startup_trans':			        False,                  # TODO: Disabling transient startup -> ssc not yet configured to start/stop calculations in the middle of startup with this model
	'is_rec_model_trans':			        False,                  # TODO: Disabling transient receiver model -> ssc not yet configured to store/retrieve receiver temperature profiles
    'cycle_type':                           'user_defined',         # 'user-defined', 'sliding', or 'fixed'
}


## Put code here in CaseStudy that will be in mediation.
class Revenue:

    @staticmethod
    def calculate_revenue(start_date, sim_days, P_out_net, params, data):
        """
        Inputs:
            start_date
            sim_days
            P_out_net
            time_steps_per_hour     (ssc_time_steps_per_hour)
            avg_price
            avg_purchase_price
            price_data

        Outputs:
            revenue
        """
        nph = int(params['time_steps_per_hour'])
        ndays = sim_days
        start = start_date
        startpt = int(util.get_time_of_year(start)/3600) * nph   # First point in annual arrays         
        price = np.array(data['dispatch_factors_ts'][startpt:int(startpt+ndays*24*nph)])
        mult = price / price.mean()   # Pricing multipliers
        
        net_gen = P_out_net
        inds_sell = np.where(net_gen > 0.0)[0]
        inds_buy = np.where(net_gen < 0.0)[0]
        rev = (net_gen[inds_sell] * mult[inds_sell] * params['avg_price']).sum() * (1./params['time_steps_per_hour'])   # Revenue from sales ($)
        rev += (net_gen[inds_buy] * mult[inds_buy] * params['avg_purchase_price']).sum() * (1./params['time_steps_per_hour']) # Electricity purchases ($)
        return rev

    
    # Calculate penalty for missing day-ahead schedule (assuming day-ahead schedule step is 1-hour for now)
    @staticmethod
    def calculate_day_ahead_penalty(sim_days, schedules, P_out_net, params, disp_soln_tracking, 
        disp_params_tracking, disp_net_electrical_output=None):
        """
        Inputs:
            sim_days
            time_steps_per_hour                     ssc_time_steps_per_hour
            schedules
            P_out_net
            disp_net_electrical_output
            disp_soln_tracking
            disp_params_tracking
            day_ahead_diff
            day_ahead_ignore_off
            day_ahead_tol_plus
            day_ahead_tol_minus
            day_ahead_pen_plus
            day_ahead_pen_minus

        Outputs:
            day_ahead_diff
            day_ahead_penalty
            day_ahead_diff_over_tol_plus
            day_ahead_diff_over_tol_minus
            day_ahead_diff_ssc_disp_gross
            day_ahead_penalty_tot
            day_ahead_diff_tot
        """


        ndays = max(1, sim_days)
        nph = int(params['time_steps_per_hour'])

        day_ahead_diff = {k:np.zeros((ndays, 24)) for k in ['ssc', 'disp', 'disp_raw']}
        day_ahead_penalty = {k:np.zeros((ndays, 24)) for k in ['ssc', 'disp', 'disp_raw']}
        day_ahead_diff_over_tol_plus = {k:0.0 for k in ['ssc', 'disp', 'disp_raw']}  
        day_ahead_diff_over_tol_minus = {k:0.0 for k in ['ssc', 'disp', 'disp_raw']}
        
        day_ahead_diff_ssc_disp_gross = np.zeros((ndays, 24))
                
        # Calculate penalty from ssc or dispatch results (translated to ssc time steps)
        for d in range(ndays):
            if len(schedules) > d and schedules[d] is not None:  # Schedule exists
                for j in range(24):  # Hours per day
                    target = schedules[d][j]       # Target generation during the schedule step
                    p = d*24*nph + j*nph                # First point in result arrays from ssc solutions
                    
                    wnet = {k:0.0 for k in ['ssc', 'disp', 'disp_raw']}
                    wnet['ssc'] = P_out_net[p:p+nph].sum() * 1./nph                             # Total generation from ssc during the schedule step (MWhe)
                    if disp_net_electrical_output is not None:
                        wnet['disp'] = disp_net_electrical_output[p:p+nph].sum() * 1./nph * 1.e-3   # Total generation from dispatch solution during the schedule step (MWhe)
                    
                    day_ahead_diff_ssc_disp_gross[d,j] = wnet['ssc'] - wnet['disp']
                    
                    # Calculate generation directly from dispatch schedule before interpolation
                    if len(disp_soln_tracking)>0:  # Dispatch solutions were saved
                        i = d*24+j
                        delta_e = disp_params_tracking[i].Delta_e
                        delta = disp_params_tracking[i].Delta 
                        wdisp = disp_soln_tracking[i].net_electrical_output/1000.  # Net energy sold to grid (MWe)
                        inds = np.where(np.array(delta_e) <= 1.0)[0]
                        wnet['disp_raw'] = sum([wdisp[i]*delta[i] for i in inds])  # MWhe cumulative generation 
                        
                    for k in day_ahead_diff.keys():
                        day_ahead_diff[k][d,j] = wnet[k] - target
                        
                        if not params['day_ahead_ignore_off'] or target>0.0 or wnet[k]>0:  # Enforce penalties for missing schedule
                            day_ahead_diff[k][d,j] = wnet[k] - target
                            if day_ahead_diff[k][d,j] > params['day_ahead_tol_plus']:
                                day_ahead_penalty[k][d,j] = day_ahead_diff[k][d,j] * params['day_ahead_pen_plus']
                                day_ahead_diff_over_tol_plus[k] += day_ahead_diff[k][d,j]
                            elif day_ahead_diff[k][d,j] < params['day_ahead_tol_minus']:
                                day_ahead_penalty[k][d,j] = (-day_ahead_diff[k][d,j]) * params['day_ahead_pen_minus']
                                day_ahead_diff_over_tol_minus[k] += day_ahead_diff[k][d,j]
                                
        day_ahead_penalty_tot = {k:day_ahead_penalty[k].sum() for k in day_ahead_diff.keys()}  # Total penalty ($)
        day_ahead_diff_tot = {k:day_ahead_diff[k].sum() for k in day_ahead_diff.keys()}

        outputs = {
            'day_ahead_diff': day_ahead_diff,
            'day_ahead_penalty': day_ahead_penalty,

            'day_ahead_penalty_tot': day_ahead_penalty_tot,
            'day_ahead_diff_tot': day_ahead_diff_tot,
            'day_ahead_diff_over_tol_plus': day_ahead_diff_over_tol_plus,
            'day_ahead_diff_over_tol_minus': day_ahead_diff_over_tol_minus
        }
        return outputs

    
    @staticmethod
    def calculate_startup_ramping_penalty(plant_design, q_startup, Q_thermal, q_pb, P_cycle, q_dot_pc_startup, params):
        """
        Inputs:
            q_startup
            Q_thermal
            q_pb
            P_cycle
            q_dot_pc_startup
            Crsu
            Ccsu
            C_delta_w

        Outputs:
            n_starts_rec
            n_starts_rec_attempted
            n_starts_cycle
            n_starts_cycle_attempted
            cycle_ramp_up
            cycle_ramp_down
            startup_ramping_penalty
        """

        
        def find_starts(q_start, q_on):
            n = len(q_start)
            n_starts, n_start_attempts_completed = [0, 0]
            for j in range(1,n):
                start_attempt_completed = False
                if q_start[j] < 1. and q_start[j-1] >= 1.:
                    start_attempt_completed = True
                    n_start_attempts_completed +=1
                
                if start_attempt_completed and (q_on[j] > 1. and q_on[j-1] < 1.e-3):
                    n_starts += 1
                        
            return n_starts, n_start_attempts_completed

        n_starts_rec, n_starts_rec_attempted = find_starts(q_startup, Q_thermal)

        qpb_on = q_pb   # Cycle thermal power includes startup
        inds_off = np.where(P_cycle<1.e-3)[0]
        qpb_on[inds_off] = 0.0
        n_starts_cycle, n_starts_cycle_attempted = find_starts(q_dot_pc_startup, qpb_on)

        n = len(P_cycle)
        cycle_ramp_up = 0.0
        cycle_ramp_down = 0.0
        w = P_cycle
        for j in range(1,n):
            diff =  w[j] - w[j-1]
            if diff > 0:
                cycle_ramp_up += diff
            elif diff < 0:
                cycle_ramp_down += (-diff)
        
        startup_ramping_penalty = n_starts_rec*plant_design['Crsu'] + n_starts_cycle*plant_design['Ccsu'] 
        startup_ramping_penalty += cycle_ramp_up*plant_design['C_delta_w']*1000 + cycle_ramp_down*plant_design['C_delta_w']*1000

        outputs = {
            'n_starts_rec': n_starts_rec,
            'n_starts_rec_attempted': n_starts_rec_attempted,
            'n_starts_cycle': n_starts_cycle,
            'n_starts_cycle_attempted': n_starts_cycle_attempted,
            'cycle_ramp_up': cycle_ramp_up,
            'cycle_ramp_down': cycle_ramp_down,
            'startup_ramping_penalty': startup_ramping_penalty
        }
        return outputs

class CaseStudy:   

    #--- Simulate flux maps
    @staticmethod
    def simulate_flux_maps(plant_design, ssc_time_steps_per_hour, ground_truth_weather_data):
        """
        Outputs:
            A_sf_in
            eta_map
            flux_maps
        """

        print ('Simulating flux maps')
        start = timeit.default_timer()
        D = plant_design.copy()
        D['time_steps_per_hour'] = ssc_time_steps_per_hour
        D['solar_resource_data'] = ground_truth_weather_data
        D['time_start'] = 0.0
        D['time_stop'] = 1.0*3600  
        D['field_model_type'] = 2
        # if self.is_debug:
        #     D['delta_flux_hrs'] = 4
        #     D['n_flux_days'] = 2
        R, state = ssc_wrapper.call_ssc(D, ['eta_map_out', 'flux_maps_for_import', 'A_sf'])
        print('Time to simulate flux maps = %.2fs'%(timeit.default_timer() - start))
        # return flux_maps
        A_sf_in = R['A_sf']
        eta_map = R['eta_map_out']
        flux_maps = [x[2:] for x in R['flux_maps_for_import']]
        return {'A_sf_in': A_sf_in, 'eta_map': eta_map, 'flux_maps': flux_maps}


    @staticmethod
    def reupdate_ssc_constants(D, params, data):
        D['solar_resource_data'] = data['solar_resource_data']
        D['dispatch_factors_ts'] = data['dispatch_factors_ts']

        D['ppa_multiplier_model'] = params['ppa_multiplier_model']
        D['time_steps_per_hour'] = params['time_steps_per_hour']
        D['is_rec_model_trans'] = params['is_rec_model_trans']
        D['is_rec_startup_trans'] = params['is_rec_startup_trans']
        D['rec_control_per_path'] = params['rec_control_per_path']
        D['field_model_type'] = params['field_model_type']
        D['eta_map_aod_format'] = params['eta_map_aod_format']
        D['is_rec_to_coldtank_allowed'] = params['is_rec_to_coldtank_allowed']
        D['is_dispatch'] = params['is_dispatch']
        D['is_dispatch_targets'] = params['is_dispatch_targets']

        #--- Set field control parameters
        if params['control_field'] == 'CD_data':
            D['rec_su_delay'] = params['rec_su_delay']
            D['rec_qf_delay'] = params['rec_qf_delay']

        #--- Set receiver control parameters
        if params['control_receiver'] == 'CD_data':
            D['is_rec_user_mflow'] = params['is_rec_user_mflow']
            D['rec_su_delay'] = params['rec_su_delay']
            D['rec_qf_delay'] = params['rec_qf_delay']
        elif params['control_receiver'] == 'ssc_clearsky':
            D['rec_clearsky_fraction'] = params['rec_clearsky_fraction']
            D['rec_clearsky_model'] = params['rec_clearsky_model']
            D['rec_clearsky_dni'] = data['clearsky_data'].tolist()
        elif params['control_receiver'] == 'ssc_actual_dni':
            D['rec_clearsky_fraction'] = params['rec_clearsky_fraction']

        return


    @staticmethod
    def default_ssc_return_vars():
        return ['beam', 'clearsky', 'tdry', 'wspd', 'solzen', 'solaz', 'pricing_mult',
                   'sf_adjust_out', 'q_sf_inc', 'eta_field', 'defocus', 
                   'q_dot_rec_inc', 'Q_thermal', 'm_dot_rec', 'q_startup', 'q_piping_losses', 'q_thermal_loss', 'eta_therm',
                   'T_rec_in', 'T_rec_out', 'T_panel_out',
                   'T_tes_hot', 'T_tes_cold', 'mass_tes_cold', 'mass_tes_hot', 'q_dc_tes', 'q_ch_tes', 'e_ch_tes', 'tank_losses', 'hot_tank_htf_percent_final',
                   'm_dot_cr_to_tes_hot', 'm_dot_tes_hot_out', 'm_dot_pc_to_tes_cold', 'm_dot_tes_cold_out', 'm_dot_field_to_cycle', 'm_dot_cycle_to_field',
                   'P_cycle','eta', 'T_pc_in', 'T_pc_out', 'q_pb', 'q_dot_pc_startup', 'P_out_net', 
                   'P_tower_pump', 'htf_pump_power', 'P_cooling_tower_tot', 'P_fixed', 'P_plant_balance_tot', 'P_rec_heattrace', 'q_heater', 'P_cycle_off_heat', 'pparasi',
                   'is_rec_su_allowed', 'is_pc_su_allowed', 'is_pc_sb_allowed',
                   'op_mode_1', 'op_mode_2', 'op_mode_3', 'q_dot_est_cr_on', 'q_dot_est_cr_su', 'q_dot_est_tes_dc', 'q_dot_est_tes_ch', 'q_dot_pc_target_on'
                   ]


    @staticmethod
    def default_disp_stored_vars():
        return ['cycle_on', 'cycle_standby', 'cycle_startup', 'receiver_on', 'receiver_startup', 'receiver_standby', 
               'receiver_power', 'thermal_input_to_cycle', 'electrical_output_from_cycle', 'net_electrical_output', 'tes_soc',
               'yrsd', 'ursd']
    

    @staticmethod
    def get_user_flow_paths(flow_path1_file, flow_path2_file, time_steps_per_hour, helio_reflectance, use_measured_reflectivity,
        soiling_avail=None, fixed_soiling_loss=None):
        """Load user flow paths into the ssc input dict (D)"""
        flow_path1_data = np.genfromtxt(flow_path1_file)
        flow_path2_data = np.genfromtxt(flow_path2_file)
        if time_steps_per_hour != 60:
            flow_path1_data = np.array(util.translate_to_new_timestep(flow_path1_data, 1./60, 1./time_steps_per_hour))
            flow_path2_data = np.array(util.translate_to_new_timestep(flow_path2_data, 1./60, 1./time_steps_per_hour))
        mult = np.ones_like(flow_path1_data)
        if not use_measured_reflectivity:                                                   # Scale mass flow based on simulated reflectivity vs. CD actual reflectivity
            rho = (1-fixed_soiling_loss) * helio_reflectance                                # Simulated heliostat reflectivity
            mult = rho*np.ones(365) / (soiling_avail*helio_reflectance)                     # Ratio of simulated / CD reflectivity
            mult = np.repeat(mult, 24*time_steps_per_hour)                                  # Annual array at ssc resolution
        rec_user_mflow_path_1 = (flow_path1_data * mult).tolist()                          # Note ssc path numbers are reversed relative to CD path numbers
        rec_user_mflow_path_2 = (flow_path2_data * mult).tolist()
        return rec_user_mflow_path_1, rec_user_mflow_path_2





########################################################################################################################################################
if __name__ == '__main__':
    start = timeit.default_timer()
    os.chdir(os.path.dirname(__file__))
    
    # Mediator inputs and parameters
    start_date = datetime.datetime(2018, 10, 14)
    sim_days = 1
    m_vars = mediator_params.copy()
    d_vars = dispatch.dispatch_wrap_params.copy()
    timestep_days = d_vars['dispatch_frequency']/24.
    ssc_time_steps_per_hour = m_vars['time_steps_per_hour']
    disp_soln_tracking = []
    disp_params_tracking = []


    # Data - get historical weather
    ground_truth_weather_data = util.read_weather_data(m_vars['ground_truth_weather_file'])
    if ssc_time_steps_per_hour != 60:
        ground_truth_weather_data = util.update_weather_timestep(ground_truth_weather_data, ssc_time_steps_per_hour)

    # Setup plant including calculating flux maps
    plant = plant_.Plant(design=plant_.plant_design, initial_state=plant_.plant_initial_state)   # Default parameters contain best representation of CD plant and dispatch properties
    if plant.flux_maps['A_sf_in'] == 0.0:
        plant.flux_maps = CaseStudy.simulate_flux_maps(
            plant_design = plant.design,
            ssc_time_steps_per_hour = ssc_time_steps_per_hour,
            ground_truth_weather_data = ground_truth_weather_data
            )
        assert math.isclose(plant.flux_maps['A_sf_in'], 1172997, rel_tol=1e-4)
        assert math.isclose(np.sum(plant.flux_maps['eta_map']), 10385.8, rel_tol=1e-4)
        assert math.isclose(np.sum(plant.flux_maps['flux_maps']), 44.0, rel_tol=1e-4)

    # Data - get field availability adjustment
    sf_adjust_hourly = util.get_field_availability_adjustment(ssc_time_steps_per_hour, start_date.year, m_vars['control_field'],
            m_vars['use_CD_measured_reflectivity'], plant.design, m_vars['fixed_soiling_loss'])

    # Data - get clear-sky DNI annual arrays 
    clearsky_data = np.genfromtxt(m_vars['clearsky_file'])
    if ssc_time_steps_per_hour != 60:
        clearsky_data = np.array(util.translate_to_new_timestep(clearsky_data, 1./60, 1./ssc_time_steps_per_hour))

    # Data - get receiver mass flow annual arrays
    rec_user_mflow_path_1, rec_user_mflow_path_2 = CaseStudy.get_user_flow_paths(
        flow_path1_file=m_vars['CD_mflow_path2_file'],          # Note ssc path numbers are reversed relative to CD path numbers
        flow_path2_file=m_vars['CD_mflow_path1_file'],
        time_steps_per_hour=ssc_time_steps_per_hour,
        helio_reflectance=plant.design['helio_reflectance'],
        use_measured_reflectivity=m_vars['use_CD_measured_reflectivity'],
        soiling_avail=util.get_CD_soiling_availability(start_date.year, plant.design['helio_reflectance'] * 100), # CD soiled / clean reflectivity (daily array),
        fixed_soiling_loss=m_vars['fixed_soiling_loss']
        )

    # Data - get prices
    price_multipliers = np.genfromtxt(m_vars['price_multiplier_file'])
    if m_vars['price_steps_per_hour'] != ssc_time_steps_per_hour:
        price_multipliers = util.translate_to_new_timestep(price_multipliers, 1./m_vars['price_steps_per_hour'], 1./ssc_time_steps_per_hour)
    pmavg = sum(price_multipliers)/len(price_multipliers)  
    price_data = [m_vars['avg_price']*p/pmavg  for p in price_multipliers]  # Electricity price at ssc time steps ($/MWh)

    data = {
        'sf_adjust:hourly':                 sf_adjust_hourly,
        'dispatch_factors_ts':              price_data,
        'clearsky_data':                    clearsky_data,
        'solar_resource_data':              ground_truth_weather_data,
        'rec_user_mflow_path_1':            rec_user_mflow_path_1,
        'rec_user_mflow_path_2':            rec_user_mflow_path_2,
    }


    # Dispatch inputs
    ursd_last = None
    yrsd_last = None
    weather_data_for_dispatch = None
    current_day_schedule = None
    next_day_schedule = None
    current_forecast_weather_data = None
    schedules = None
    horizon = sim_days*24*3600          # [s]
    initial_plant_state = util.get_initial_state_from_CD_data(start_date, m_vars['CD_raw_data_direc'], m_vars['CD_processed_data_direc'], plant.design)


    # Run models
    outputs_total = {}
    nupdate = int(sim_days*24 / d_vars['dispatch_frequency'])
    params = m_vars.copy()
    params.update(d_vars)                   # combine mediator and dispatch params
    params['start_date'] = start_date       # needed for initializing schedules
    for j in range(nupdate):
        dispatch_wrap = DispatchWrap(plant=plant, params=params, data=data)

        # Run dispatch model
        dispatch_outputs = dispatch_wrap.run(
            start_date=start_date,
            timestep_days=timestep_days,
            horizon=horizon,
            retvars=CaseStudy.default_disp_stored_vars(),
            ursd_last=ursd_last,
            yrsd_last=yrsd_last,
            current_forecast_weather_data=current_forecast_weather_data,
            weather_data_for_dispatch=weather_data_for_dispatch,
            schedules=schedules,
            current_day_schedule=current_day_schedule,
            next_day_schedule=next_day_schedule,
            initial_plant_state=initial_plant_state
            )

        # Run ssc model:
        D2 = plant.design.copy()            # Start compiling ssc input dict (D2)
        D2.update(plant.state)
        D2.update(plant.flux_maps)
        D2['time_start'] = int(util.get_time_of_year(start_date))
        D2['time_stop'] = util.get_time_of_year(start_date.replace(hour=0, minute=0, second=0)) + timestep_days*24*3600
        D2['sf_adjust:hourly'] = data['sf_adjust:hourly']
        CaseStudy.reupdate_ssc_constants(D2, m_vars, data)
        if m_vars['control_receiver'] == 'CD_data':
            D2['rec_user_mflow_path_1'] = data['rec_user_mflow_path_1']
            D2['rec_user_mflow_path_2'] = data['rec_user_mflow_path_2']

        if m_vars['is_optimize'] and dispatch_outputs['Rdisp'] is not None:
            D2.update(vars(dispatch_outputs['ssc_dispatch_targets']))

        D2['time_stop'] = int(util.get_time_of_year(start_date)) + int(d_vars['dispatch_frequency']*3600)
        retvars = CaseStudy.default_ssc_return_vars()
        napply = int(ssc_time_steps_per_hour*d_vars['dispatch_frequency'])                   # Number of ssc time points accepted after each solution 
        results, new_plant_state = ssc_wrapper.call_ssc(D2, retvars, plant_state_pt = napply-1, npts = napply)
        
        # Update saved plant state, post model run
        persistance_vars = plant_.Plant.update_persistence(
            plant.state,
            results,
            new_plant_state['rec_op_mode_initial'],
            new_plant_state['pc_op_mode_initial'],
            1./ssc_time_steps_per_hour)
        new_plant_state.update(persistance_vars)
        plant.state.update(new_plant_state)

        if start_date == datetime.datetime(2018, 10, 14, 0, 0):
            assert math.isclose(plant.state['pc_startup_energy_remain_initial'], 29339.9, rel_tol=1e-4)
            assert math.isclose(plant.state['pc_startup_time_remain_init'], 0.5, rel_tol=1e-4)
            assert math.isclose(plant.state['rec_startup_energy_remain_init'], 141250000, rel_tol=1e-4)
            assert math.isclose(plant.state['rec_startup_time_remain_init'], 1.15, rel_tol=1e-4)
            assert math.isclose(plant.state['disp_rec_persist0'], 1001, rel_tol=1e-4)
            assert math.isclose(plant.state['disp_rec_off0'], 1001, rel_tol=1e-4)
            assert math.isclose(plant.state['disp_pc_persist0'], 1001, rel_tol=1e-4)
            assert math.isclose(plant.state['disp_pc_off0'], 1001, rel_tol=1e-4)

        if start_date == datetime.datetime(2018, 10, 14, 1, 0):
            assert math.isclose(plant.state['pc_startup_energy_remain_initial'], 29339.9, rel_tol=1e-4)
            assert math.isclose(plant.state['pc_startup_time_remain_init'], 0.5, rel_tol=1e-4)
            assert math.isclose(plant.state['rec_startup_energy_remain_init'], 141250000, rel_tol=1e-4)
            assert math.isclose(plant.state['rec_startup_time_remain_init'], 1.15, rel_tol=1e-4)
            assert math.isclose(plant.state['disp_rec_persist0'], 1002, rel_tol=1e-4)
            assert math.isclose(plant.state['disp_rec_off0'], 1002, rel_tol=1e-4)
            assert math.isclose(plant.state['disp_pc_persist0'], 1002, rel_tol=1e-4)
            assert math.isclose(plant.state['disp_pc_off0'], 1002, rel_tol=1e-4)
        
        results.update(dispatch_outputs['Rdisp'])         # add dispatch results to ssc results

        # Calculate post-simulation financials
        revenue = Revenue.calculate_revenue(start_date, timestep_days, results['P_out_net'], m_vars, data)
        day_ahead_penalties = Revenue.calculate_day_ahead_penalty(timestep_days, dispatch_outputs['schedules'],
            results['P_out_net'], m_vars, disp_soln_tracking, disp_params_tracking)
        startup_ramping_penalties = Revenue.calculate_startup_ramping_penalty(
            plant_design=plant.design,
            q_startup=results['q_startup'],
            Q_thermal=results['Q_thermal'],
            q_pb=results['q_pb'],
            P_cycle=results['P_cycle'],
            q_dot_pc_startup=results['q_dot_pc_startup'],
            params=m_vars
        )        
        
        # Update inputs for next call
        schedules = dispatch_outputs['schedules']
        current_day_schedule = dispatch_outputs['current_day_schedule']
        next_day_schedule = dispatch_outputs['next_day_schedule']
        start_date += datetime.timedelta(hours=d_vars['dispatch_frequency'])
        horizon -= int(timestep_days*24*3600)
        ursd_last = dispatch_outputs['ursd_last']
        yrsd_last = dispatch_outputs['yrsd_last']
        current_forecast_weather_data = dispatch_outputs['current_forecast_weather_data']
        weather_data_for_dispatch = dispatch_outputs['weather_data_for_dispatch']
        initial_plant_state = plant.state

        # Aggregate totals
        outputs = {
            'revenue': revenue,
            'Q_thermal': results['Q_thermal'],
            'P_cycle': results['P_cycle'],
            'P_out_net': results['P_out_net']
        }
        outputs.update(day_ahead_penalties)
        outputs.update(startup_ramping_penalties)
        outputs['total_receiver_thermal'] = outputs.pop('Q_thermal').sum() * 1.e-3 * (1./ssc_time_steps_per_hour)
        outputs['total_cycle_gross'] = outputs.pop('P_cycle').sum() * 1.e-3 * (1./ssc_time_steps_per_hour)
        outputs['total_cycle_net'] = outputs.pop('P_out_net').sum() * 1.e-3 * (1./ssc_time_steps_per_hour)
        if not outputs_total:   # if empty
            outputs_total = outputs.copy()
        else:
            for key in outputs:
                if isinstance(outputs[key], dict):
                    for key_inner in outputs[key]:
                        outputs_total[key][key_inner] += outputs[key][key_inner]
                else:
                    outputs_total[key] += outputs[key]
    

    elapsed = timeit.default_timer() - start

    # All of these outputs are just sums of the individual calls
    print ('Total time elapsed = %.2fs'%(timeit.default_timer() - start))
    print ('Receiver thermal generation = %.5f GWht'%outputs_total['total_receiver_thermal'])
    print ('Cycle gross generation = %.5f GWhe'%outputs_total['total_cycle_gross'])
    print ('Cycle net generation = %.5f GWhe'%outputs_total['total_cycle_net'])
    print ('Receiver starts = %d completed, %d attempted'%(outputs_total['n_starts_rec'], outputs_total['n_starts_rec_attempted']))
    print ('Cycle starts = %d completed, %d attempted'%(outputs_total['n_starts_cycle'], outputs_total['n_starts_cycle_attempted']))
    print ('Cycle ramp-up = %.3f'%outputs_total['cycle_ramp_up'])
    print ('Cycle ramp-down = %.3f'%outputs_total['cycle_ramp_down'])
    print ('Total under-generation from schedule (beyond tolerance) = %.3f MWhe (ssc), %.3f MWhe (dispatch)'
        %(outputs_total['day_ahead_diff_over_tol_minus']['ssc'], outputs_total['day_ahead_diff_over_tol_minus']['disp']))
    print ('Total over-generation from schedule  (beyond tolerance)  = %.3f MWhe (ssc), %.3f MWhe (dispatch)'
        %(outputs_total['day_ahead_diff_over_tol_plus']['ssc'], outputs_total['day_ahead_diff_over_tol_plus']['disp']))
    print ('Revenue = $%.2f'%outputs_total['revenue'])
    print ('Day-ahead schedule penalty = $%.2f (ssc), $%.2f (dispatch)'%(outputs_total['day_ahead_penalty_tot']['ssc'], outputs_total['day_ahead_penalty_tot']['disp']))
    print ('Startup/ramping penalty = $%.2f'%outputs_total['startup_ramping_penalty'])

    # Basic regression tests for refactoring
    assert math.isclose(outputs_total['total_receiver_thermal'], 3.85, rel_tol=1e-3)
    assert math.isclose(outputs_total['total_cycle_gross'], 1.89, rel_tol=1e-2)
    assert math.isclose(outputs_total['total_cycle_net'], 1.74, rel_tol=1e-2)
    assert math.isclose(outputs_total['cycle_ramp_up'], 120.2, rel_tol=1e-3)
    assert math.isclose(outputs_total['cycle_ramp_down'], 121.0, rel_tol=1e-3)
    assert math.isclose(outputs_total['revenue'], 243565, rel_tol=1e-2)
    assert math.isclose(outputs_total['startup_ramping_penalty'], 7200, rel_tol=1e-3)

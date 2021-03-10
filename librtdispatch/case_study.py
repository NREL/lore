import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
from csv import reader
from copy import deepcopy
import timeit
import pandas
import os
import math
import datetime

import util
import dispatch
import ssc_wrapper
import loredash.mediation.plant as plant_


mediator_params = {
    # Not yet categorized:
	'ppa_multiplier_model':			        1,
	'rec_control_per_path':			        True,
	'field_model_type':				        3,
	'eta_map_aod_format':			        False,
	'is_dispatch':					        0,                      # Always disable dispatch optimization in ssc
	'is_dispatch_targets':			        True,		            # True if (is_optimize or control_cycle == 'CD_data')
	'is_rec_user_mflow':			        False,		            # or this should be unassigned, True if control_receiver == 'CD_data'
	'rec_clearsky_fraction':		        1.0,
    'rec_clearsky_model':                   0,
    'rec_su_delay':                         0.,                     # = 0.01 if control_field == 'CD_data' or control_receiver == 'CD_data'  Set receiver start time and energy to near zero to enforce CD receiver startup timing
    'rec_qf_delay':                         0.,                     # = 0.01 if control_field == 'CD_data' or control_receiver == 'CD_data'
    'is_rec_to_coldtank_allowed':           True,

    # Control conditions
	'time_steps_per_hour':			        60,			            # Simulation time resolution in ssc (1min)   DUPLICATED to: ssc_time_steps_per_hour
    'is_optimize':					        True,                   # Use dispatch optimziation
	'control_field':				        'ssc',                  #'CD_data' = use CD data to control heliostats tracking, heliostats offline, and heliostat daily reflectivity.  Receiver startup time is set to zero so that simulated receiver starts when CD tracking begins
                                                                    #'ssc' = allow ssc to control heliostat field operations, assuming all heliostats are available

	'control_receiver':				        'ssc_clearsky',         #'CD_data' = use CD data to control receiver mass flow. Receiver startup time is set to zero so that simulated receiver starts when CD receiver finishes startup
                                                                    #'ssc_clearsky' = use expected clearsky DNI to control receiver mass flow.  If field_control = 'ssc' then the historical median startup time from CD data will be used to control receiver startup
                                                                    #'ssc_actual_dni' = use actual DNI to control receiver mass flow.  If field_control = 'ssc' then the historical median startup time from CD data will be used to control receiver startup   

	'control_cycle':				        'ssc_heuristic',        # Only used if is_optimize = False
                                                                    # 'CD_data' = use CD actual cycle operation to control cycle dispatch targets
                                                                    # 'ssc_heuristic' = allow ssc heuristic (no consideration of TOD price) to control cycle dispatch

    # Dispatch optimization
	'dispatch_frequency':			        1,                      # Frequency of dispatch re-optimization (hr)
	'dispatch_weather_horizon':		        2,                      # Time point in hours (relative to start of optimization horizon) defining the transition from actual weather to forecasted weather used in the dispatch model. Set to 0 to use forecasts for the full horizon, set to -1 to use actual weather for the full horizon, or any value > 0 to combine actual/forecasted weather
    'dispatch_horizon':                     48.,                    # Dispatch time horizon (hr) 
    'dispatch_horizon_update':              24.,                    # Frequency of dispatch time horizon update (hr) -> set to the same value as dispatch_frequency for a fixed-length horizon 
    'dispatch_steplength_array':            [5, 15, 60],            # Dispatch time step sizes (min)
    'dispatch_steplength_end_time':         [1, 4, 48],             # End time for dispatch step lengths (hr)
    'nonlinear_model_time':                 4.0,                    # Amount of time to apply nonlinear dispatch model (hr) (not currently used)
    'disp_time_weighting':                  0.999,                  # Dispatch time weighting factor. 
    'use_linear_dispatch_at_night':         False,                  # Revert to the linear dispatch model when all of the time-horizon in the nonlinear model is at night.
    'night_clearsky_cutoff':                100.,                   # Cutoff value for clear-sky DNI defining "night"

    # Weather forecasts
	'forecast_issue_time':			        16,                     # Time at which weather forecast is issued (hr, 0-23), assumed to be in standard time.  Forecasts issued at midnight UTC 
	'forecast_steps_per_hour':		        1,                      # Number of time steps per hour in weather forecasts
    'forecast_update_frequency':            24,                     # Weather forecast update interval (hr)

    # Price
	'price_steps_per_hour':			        1,                      # Number of steps per hour in electricity price multipliers
	'avg_price':					        138,                    # Average electricity price ($/MWh):  CD original PPA was $138/MWh
    'avg_purchase_price':                   30,                     # Average electricity purchase price ($/MWh) (note, this isn't currently used in the dispatch model)
    'avg_price_disp_storage_incentive':     0.0,                    # Average electricity price ($/MWh) used in dispatch model storage inventory incentive

    # Day-ahead schedule targets
	'use_day_ahead_schedule':		        True,                   # Use day-ahead generation targets
	'day_ahead_schedule_from':		        'calculated',           # 'calculated' = calculate day-ahead schedule during solution, 'NVE'= use NVE-provided schedule for CD
	'day_ahead_schedule_time':		        10,                     # Time of day at which day-ahead schedule is due (hr, 0-23), assumed to be in standard time
	'day_ahead_schedule_steps_per_hour':    1,                      # Time resolution of day-ahead schedule
    'day_ahead_pen_plus':                   500,                    # Penalty for over-generation relative to day-ahead schedule ($/MWhe)
    'day_ahead_pen_minus':                  500,                    # Penalty for under-generation relative to day-ahead schedule ($/MWhe)
    'day_ahead_tol_plus':                   5,                      # Tolerance for over-production relative to day-ahead schedule before incurring penalty (MWhe)
    'day_ahead_tol_minus':                  5,                      # Tolerance for under-production relative to day-ahead schedule before incurring penalty (MWhe)
    'day_ahead_ignore_off':                 True,                   # Don't apply schedule penalties when cycle is scheduled to be off for the full hour (MWhe)

    # Field, receiver, and cycle simulation options
    'use_CD_measured_reflectivity':	        False,                  # Use measured heliostat reflectivity from CD data
    'fixed_soiling_loss':			        0.02,                   # Fixed soiling loss (if not using CD measured reflectivity) = 1 - (reflectivity / clean_reflectivity)
	'is_rec_startup_trans':			        False,                  # TODO: Disabling transient startup -> ssc not yet configured to start/stop calculations in the middle of startup with this model
	'is_rec_model_trans':			        False,                  # TODO: Disabling transient receiver model -> ssc not yet configured to store/retrieve receiver temperature profiles
    'cycle_type':                           'user_defined',         # 'user-defined', 'sliding', or 'fixed'

    # Miscellaneous
	'store_full_dispatch_solns':	        False,                  # Store full dispatch input parameters and solutions for each call to the dispatch model
    'force_rolling_horizon':                False,                  # Force simulation using ssc heuristic dispatch to follow a rolling horizon for debugging?
}

## Put code here in CaseStudy that will be in mediation.
class CaseStudy:
    def __init__(self, plant, params, data):

        ## DISPATCH INPUTS ###############################################################################################################################
        # Input data files: weather, masslow, clearsky DNI must have length of full annual array based on ssc time step size
        #--- Simulation start point and duration
        self.start_date = None
        self.sim_days = None       
        # self.plant = None                               # Plant design and operating properties
        
        self.user_defined_cycle_input_file = 'udpc_noTamb_dependency.csv'  # Only required if cycle_type is user_defined

        ## DISPATCH PERSISTING INTERMEDIARIES ############################################################################################################
        self.dispatch_params = dispatch.DispatchParams() # Structure to contain all inputs for dispatch model 
        self.current_time = 0                           # Current time (tracked in standard time, not local time)
        self.is_initialized = False                     # Has solution already been initalized?        
        self.CD_data_for_plotting = {}                  # Only used if control_cycle = 'CD_data' 


        ## DISPATCH OUTPUTS FOR INPUT INTO SSC ###########################################################################################################
        # see: ssc_dispatch_targets, which is a dispatch.DispatchTargets object


        ## SSC OUTPUTS ###################################################################################################################################
        self.results = None                             # Results

        self.current_day_schedule = []                  # Committed day-ahead generation schedule for current day (MWe)
        self.next_day_schedule = []                     # Predicted day-ahead generation schedule for next day (MWe)
        self.schedules = []                             # List to store all day-ahead generation schedules (MWe)
        self.weather_at_schedule = []                   # List to store weather data at the point in time the day-ahead schedule was generated
        self.disp_params_tracking = []                  # List to store dispatch parameters for each call to dispatch model
        self.disp_soln_tracking = []                    # List to store dispatch solutions for each call to dispatch model (directly from dispatch model, no translation to ssc time steps)
        self.plant_state_tracking = []                  # List to store plant state at the start of each call to the dispatch model
        self.infeasible_count = 0
        
        self.revenue = 0.0                              # Revenue over simulated days ($)
        self.startup_ramping_penalty = 0.0              # Startup and ramping penalty over all simulated days ($)
        self.day_ahead_penalty_tot = {}                 # Penalty for missing schedule over all simulated days ($)
        self.day_ahead_diff_tot = {}                    # Total difference between actual and scheduled generation (MWhe)
        self.day_ahead_diff_over_tol_plus = {}          # Total (positive) difference between actual and scheduled generation over tolerance (MWhe)
        self.day_ahead_diff_over_tol_minus = {}         # Total (negative) difference between actual and scheduled generation over tolerance (MWhe)
        
        self.day_ahead_diff = []                        # Difference between net generation and scheduled net generation (MWhe)
        self.day_ahead_penalty = []                     # Penalty for difference between net generation and scheduled net generation (MWhe)
        self.day_ahead_diff_ssc_disp_gross = []         # Difference between ssc and dispatch gross generation in schedule time steps (MWhe)
        
        self.n_starts_rec = 0                           # Number of receiver starts
        self.n_starts_rec_attempted = 0                 # Number of receiver starts, including those not completed
        self.n_starts_cycle = 0                         # Number of cycle starts
        self.n_starts_cycle_attempted = 0               # Number of cycle starts, including those not completed
        self.cycle_ramp_up = 0                          # Cycle ramp-up (MWe)
        self.cycle_ramp_down = 0                        # Cycle ramp-down (MWe)
        self.total_receiver_thermal = 0                 # Total thermal energy from receiver (GWht)
        self.total_cycle_gross = 0                      # Total gross generation by cycle (GWhe)
        self.total_cycle_net = 0                        # Total net generation by cycle (GWhe)


        self.plant = plant                              # Plant design and operating properties
        self.params = params
        self.data = data

        for key,value in params.items():
            setattr(self, key, value)

        for key,value in data.items():
            setattr(self, key, value)
        
        # Aliases (that could be combined and removed)
        self.ssc_time_steps_per_hour = params['time_steps_per_hour']
        self.use_transient_model = params['is_rec_model_trans']
        self.use_transient_startup = params['is_rec_startup_trans']
        self.ground_truth_weather_data = data['solar_resource_data']
        self.price_data = data['dispatch_factors_ts']

        # Initialize and adjust above parameters
        self.initialize()

        return


    #------------------------------------------------------------------------


    #-------------------------------------------------------------------------
    #--- Run simulation
    def run(self, start_date, timestep_days, horizon, ursd_last, yrsd_last, current_forecast_weather_data, weather_data_for_dispatch,
            schedules, current_day_schedule, next_day_schedule, initial_plant_state=None):

        time = self.current_time = self.start_date = start_date
        self.sim_days = timestep_days
        self.current_forecast_weather_data = current_forecast_weather_data
        self.weather_data_for_dispatch = weather_data_for_dispatch
        self.schedules = schedules
        self.current_day_schedule = current_day_schedule
        self.next_day_schedule = next_day_schedule
        if initial_plant_state is not None:
            self.plant.state = initial_plant_state

        # Start compiling ssc input dict (D)
        D = self.plant.design.copy()
        D.update(self.plant.state)
        D.update(self.plant.flux_maps)
        D['time_start'] = int(util.get_time_of_year(self.start_date))
        D['time_stop'] = util.get_time_of_year(self.start_date.replace(hour=0, minute=0, second=0)) + self.sim_days*24*3600
        D['sf_adjust:hourly'] = CaseStudy.get_field_availability_adjustment(self.ssc_time_steps_per_hour, self.start_date.year, self.control_field,
            self.use_CD_measured_reflectivity, self.plant.design, self.fixed_soiling_loss)
        CaseStudy.reupdate_ssc_constants(D, self.params, self.data)
        if self.control_receiver == 'CD_data':
            CaseStudy.load_user_flow_paths(
                D=D,
                flow_path_1_data=self.CD_mflow_path2_data,          # Note ssc path numbers are reversed relative to CD path numbers
                flow_path_2_data=self.CD_mflow_path1_data,
                use_measured_reflectivity=self.use_CD_measured_reflectivity,
                soiling_avail=util.get_CD_soiling_availability(self.start_date.year, D['helio_reflectance'] * 100), # CD soiled / clean reflectivity (daily array),
                fixed_soiling_loss=self.fixed_soiling_loss
                )

        #-------------------------------------------------------------------------
        # Run simulation in a rolling horizon   
        #      The ssc simulation time resolution is assumed to be <= the shortest dispatch time step
        #      All dispatch time steps and time horizons are assumed to be an integer multiple of the ssc time step
        #      Time at which the weather forecast is updated coincides with the start of an optimization interval
        #      Time at which the day-ahead generation schedule is due coincides with the start of an optimization interval

        #--- Calculate time-related values
        tod = int(util.get_time_of_day(time))                       # Current time of day (s)
        toy = int(util.get_time_of_year(time))                      # Current time of year (s)    
        start_time = util.get_time_of_year(time)                    # Time (sec) elapsed since beginning of year
        start_hour = int(start_time / 3600)                         # Time (hours) elapsed since beginning of year
        end_hour = start_hour + self.sim_days*24
        nph = int(self.ssc_time_steps_per_hour)                     # Number of time steps per hour
        total_horizon = self.sim_days*24
        ntot = int(nph*total_horizon)                               # Total number of time points in full horizon
        napply = int(nph*self.dispatch_frequency)                   # Number of ssc time points accepted after each solution 
        nupdate = int(total_horizon / self.dispatch_frequency)      # Number of update intervals
        startpt = int(start_hour*nph)                               # Start point in annual arrays
        sscstep = 3600/nph                                          # ssc time step (s)
        nominal_horizon = int(self.dispatch_horizon*3600)  
        horizon_update = int(self.dispatch_horizon_update*3600)
        freq = int(self.dispatch_frequency*3600)                    # Frequency of rolling horizon update (s)

        #--- Initialize results
        retvars = CaseStudy.default_ssc_return_vars()
        retvars_disp = self.default_disp_stored_vars()
        if self.is_optimize:
            retvars += vars(dispatch.DispatchTargets()).keys()
        R = {k:np.zeros(ntot) for k in retvars}
        for k in retvars_disp:
            R['disp_'+k] =np.zeros(ntot)

        #--- Update "forecasted" weather data (if relevant)
        if self.is_optimize and (tod == self.forecast_issue_time*3600):
            self.current_forecast_weather_data = CaseStudy.update_forecast_weather_data(
                date=time,
                current_forecast_weather_data=self.current_forecast_weather_data,
                ssc_time_steps_per_hour=self.ssc_time_steps_per_hour,
                forecast_steps_per_hour=self.forecast_steps_per_hour,
                ground_truth_weather_data=self.ground_truth_weather_data,
                forecast_issue_time=self.forecast_issue_time,
                day_ahead_schedule_time=self.day_ahead_schedule_time,
                clearsky_data=self.clearsky_data
                )

        #--- Update stored day-ahead generation schedule for current day (if relevant)
        if tod == 0 and self.use_day_ahead_schedule:
            self.next_day_schedule = [0 for s in self.next_day_schedule]

            if self.day_ahead_schedule_from == 'NVE':
                self.current_day_schedule = self.get_CD_NVE_day_ahead_schedule(time)
                self.schedules.append(self.current_day_schedule)
            
        # Don't include day-ahead schedule if one hasn't been calculated yet, or if there is no NVE schedule available on this day
        if ((toy - start_time)/3600 < 24 and self.day_ahead_schedule_from == 'calculated') or (self.day_ahead_schedule_from == 'NVE' and self.current_day_schedule == None):  
            include_day_ahead_in_dispatch = False
        else:
            include_day_ahead_in_dispatch = self.use_day_ahead_schedule

        #--- Run dispatch optimization (if relevant)
        if self.is_optimize:
            
            if start_date == datetime.datetime(2018, 10, 14, 1, 0):
                assert math.isclose(self.weather_data_for_dispatch['tz'], -8, rel_tol=1e-4)
                assert math.isclose(self.weather_data_for_dispatch['elev'], 1497.2, rel_tol=1e-4)
                assert math.isclose(self.weather_data_for_dispatch['lat'], 38.24, rel_tol=1e-4)
                assert math.isclose(self.weather_data_for_dispatch['lon'], -117.36, rel_tol=1e-4)
                assert math.isclose(sum(list(self.weather_data_for_dispatch['dn'])), 526513.8, rel_tol=1e-4)
                assert math.isclose(sum(list(self.weather_data_for_dispatch['df'])), 0, rel_tol=1e-4)
                assert math.isclose(sum(list(self.weather_data_for_dispatch['gh'])), 0, rel_tol=1e-4)
                assert math.isclose(sum(list(self.weather_data_for_dispatch['tdry'])), 10522.8, rel_tol=1e-4)

                assert math.isclose(self.ground_truth_weather_data['tz'], -8, rel_tol=1e-4)
                assert math.isclose(self.ground_truth_weather_data['elev'], 1497.2, rel_tol=1e-4)
                assert math.isclose(self.ground_truth_weather_data['lat'], 38.24, rel_tol=1e-4)
                assert math.isclose(self.ground_truth_weather_data['lon'], -117.36, rel_tol=1e-4)
                assert math.isclose(sum(list(self.ground_truth_weather_data['dn'])), 150372982.8, rel_tol=1e-4)
                assert math.isclose(sum(list(self.ground_truth_weather_data['df'])), 0, rel_tol=1e-4)
                assert math.isclose(sum(list(self.ground_truth_weather_data['gh'])), 0, rel_tol=1e-4)
                assert math.isclose(sum(list(self.ground_truth_weather_data['tdry'])), 6825483.6, rel_tol=1e-4)

                assert math.isclose(self.current_forecast_weather_data['tz'], -8, rel_tol=1e-4)
                assert math.isclose(self.current_forecast_weather_data['elev'], 1497.2, rel_tol=1e-4)
                assert math.isclose(self.current_forecast_weather_data['lat'], 38.24, rel_tol=1e-4)
                assert math.isclose(self.current_forecast_weather_data['lon'], -117.36, rel_tol=1e-4)
                assert math.isclose(sum(list(self.current_forecast_weather_data['dn'])), 1077438.6, rel_tol=1e-4)
                assert math.isclose(sum(list(self.current_forecast_weather_data['df'])), 0, rel_tol=1e-4)
                assert math.isclose(sum(list(self.current_forecast_weather_data['gh'])), 0, rel_tol=1e-4)
                assert math.isclose(sum(list(self.current_forecast_weather_data['tdry'])), 28686.6, rel_tol=1e-4)

                assert time == datetime.datetime(2018, 10, 14, 1, 0 ,0)
                assert horizon == 82800
                assert self.dispatch_weather_horizon == 2

            #--- Update weather to use in dispatch optimization for this optimization horizon
            self.weather_data_for_dispatch = dispatch.update_dispatch_weather_data(
                weather_data = self.weather_data_for_dispatch,
                replacement_real_weather_data = self.ground_truth_weather_data,
                replacement_forecast_weather_data = self.current_forecast_weather_data,
                datetime = time,
                total_horizon = horizon/3600.,
                dispatch_horizon = self.dispatch_weather_horizon
                )

            if start_date == datetime.datetime(2018, 10, 14, 0, 0):
                assert math.isclose(self.weather_data_for_dispatch['tz'], -8, rel_tol=1e-4)
                assert math.isclose(self.weather_data_for_dispatch['elev'], 1497.2, rel_tol=1e-4)
                assert math.isclose(self.weather_data_for_dispatch['lat'], 38.24, rel_tol=1e-4)
                assert math.isclose(self.weather_data_for_dispatch['lon'], -117.36, rel_tol=1e-4)
                assert math.isclose(sum(list(self.weather_data_for_dispatch['dn'])), 526513.7, rel_tol=1e-4)
                assert math.isclose(sum(list(self.weather_data_for_dispatch['df'])), 0, rel_tol=1e-4)
                assert math.isclose(sum(list(self.weather_data_for_dispatch['gh'])), 0, rel_tol=1e-4)
                assert math.isclose(sum(list(self.weather_data_for_dispatch['tdry'])), 10522.8, rel_tol=1e-4)

            if start_date == datetime.datetime(2018, 10, 14, 1, 0):
                assert math.isclose(self.weather_data_for_dispatch['tz'], -8, rel_tol=1e-4)
                assert math.isclose(self.weather_data_for_dispatch['elev'], 1497.2, rel_tol=1e-4)
                assert math.isclose(self.weather_data_for_dispatch['lat'], 38.24, rel_tol=1e-4)
                assert math.isclose(self.weather_data_for_dispatch['lon'], -117.36, rel_tol=1e-4)
                assert math.isclose(sum(list(self.weather_data_for_dispatch['dn'])), 526513.8, rel_tol=1e-4)
                assert math.isclose(sum(list(self.weather_data_for_dispatch['df'])), 0, rel_tol=1e-4)
                assert math.isclose(sum(list(self.weather_data_for_dispatch['gh'])), 0, rel_tol=1e-4)
                assert math.isclose(sum(list(self.weather_data_for_dispatch['tdry'])), 10737.5, rel_tol=1e-4)

            #--- Run ssc for dispatch estimates: (using weather forecast time resolution for weather data and specified ssc time step)
            npts_horizon = int(horizon/3600 * nph)
            R_est = dispatch.estimates_for_dispatch_model(
                plant_design = D,
                toy = toy,
                horizon = horizon,
                weather_data = self.weather_data_for_dispatch,
                N_pts_horizon = npts_horizon,
                clearsky_data = self.clearsky_data,
                start_pt = startpt
            )

            if start_date == datetime.datetime(2018, 10, 14, 0, 0):
                assert math.isclose(sum(list(R_est["Q_thermal"])), 230346, rel_tol=1e-4)
                assert math.isclose(sum(list(R_est["m_dot_rec"])), 599416, rel_tol=1e-4)
                assert math.isclose(sum(list(R_est["clearsky"])), 543582, rel_tol=1e-4)
                assert math.isclose(sum(list(R_est["P_tower_pump"])), 2460.8, rel_tol=1e-4)

            if start_date == datetime.datetime(2018, 10, 14, 1, 0):
                assert math.isclose(sum(list(R_est["Q_thermal"])), 230347, rel_tol=1e-4)
                assert math.isclose(sum(list(R_est["m_dot_rec"])), 599355, rel_tol=1e-4)
                assert math.isclose(sum(list(R_est["clearsky"])), 543582, rel_tol=1e-4)
                assert math.isclose(sum(list(R_est["P_tower_pump"])), 2460.6, rel_tol=1e-4)

            #--- Set dispatch optimization properties for this time horizon using ssc estimates
            disp_in = dispatch.setup_dispatch_model(
                R_est = R_est,
                freq = freq,
                horizon = horizon,
                include_day_ahead_in_dispatch = include_day_ahead_in_dispatch,
                dispatch_params = self.dispatch_params,
                dispatch_steplength_array = self.dispatch_steplength_array,
                dispatch_steplength_end_time = self.dispatch_steplength_end_time,
                dispatch_horizon = self.dispatch_horizon,
                plant = self.plant,
                nonlinear_model_time = self.nonlinear_model_time,
                use_linear_dispatch_at_night = self.use_linear_dispatch_at_night,
                clearsky_data = self.clearsky_data,
                night_clearky_cutoff = self.night_clearsky_cutoff,
                disp_time_weighting = self.disp_time_weighting,
                price = self.price_data[startpt:startpt+npts_horizon],          # [$/MWh] Update prices for this horizon
                sscstep = sscstep,
                avg_price = self.avg_price,
                avg_price_disp_storage_incentive = self.avg_price_disp_storage_incentive,
                avg_purchase_price = self.avg_purchase_price,
                day_ahead_tol_plus = self.day_ahead_tol_plus,
                day_ahead_tol_minus = self.day_ahead_tol_minus,
                tod = tod,
                current_day_schedule = self.current_day_schedule,
                day_ahead_pen_plus = self.day_ahead_pen_plus,
                day_ahead_pen_minus = self.day_ahead_pen_minus,
                night_clearsky_cutoff = self.night_clearsky_cutoff,
                ursd_last = ursd_last,
                yrsd_last = yrsd_last
            )

            include = {"pv": False, "battery": False, "persistence": False, "force_cycle": False, "op_assumptions": False,
                        "signal":include_day_ahead_in_dispatch, "simple_receiver": False}
                
            dispatch_soln = dispatch.run_dispatch_model(disp_in, include)

            if start_date == datetime.datetime(2018, 10, 14, 0, 0):
                assert math.isclose(sum(list(dispatch_soln.cycle_on)), 16, rel_tol=1e-4)
                assert math.isclose(sum(list(dispatch_soln.cycle_startup)), 2, rel_tol=1e-4)
                assert math.isclose(sum(list(dispatch_soln.drsu)), 2.15, rel_tol=1e-4)
                assert math.isclose(sum(list(dispatch_soln.electrical_output_from_cycle)), 1731858, rel_tol=1e-4)
                assert math.isclose(sum(list(dispatch_soln.frsu)), 1.15, rel_tol=1e-4)
                assert math.isclose(dispatch_soln.objective_value, 206946.6, rel_tol=1e-4)
                assert math.isclose(dispatch_soln.s0, 832639.4, rel_tol=1e-4)

            if start_date == datetime.datetime(2018, 10, 14, 1, 0):
                assert math.isclose(sum(list(dispatch_soln.cycle_on)), 15, rel_tol=1e-4)
                assert math.isclose(sum(list(dispatch_soln.cycle_startup)), 2, rel_tol=1e-4)
                assert math.isclose(sum(list(dispatch_soln.drsu)), 2.15, rel_tol=1e-4)
                assert math.isclose(sum(list(dispatch_soln.electrical_output_from_cycle)), 1743183, rel_tol=1e-4)
                assert math.isclose(sum(list(dispatch_soln.frsu)), 1.15, rel_tol=1e-4)
                assert math.isclose(dispatch_soln.objective_value, 208838, rel_tol=1e-4)
                assert math.isclose(dispatch_soln.s0, 832181, rel_tol=1e-4)

            if dispatch_soln is not None:
                Rdisp_all = dispatch_soln.get_solution_at_ssc_steps(self.dispatch_params, sscstep/3600., freq/3600.)
                Rdisp = {'disp_'+key:value for key,value in Rdisp_all.items() if key in retvars_disp}

                # TODO: make the time triggering more robust; shouldn't be an '==' as the program may be offline at the time or running at intervals
                #  that won't exactly hit it
                if self.use_day_ahead_schedule and self.day_ahead_schedule_from == 'calculated' and tod/3600 == self.day_ahead_schedule_time:
                    self.next_day_schedule = dispatch.get_day_ahead_schedule(
                        day_ahead_schedule_steps_per_hour = self.day_ahead_schedule_steps_per_hour,
                        Delta = self.dispatch_params.Delta,
                        Delta_e = self.dispatch_params.Delta_e,
                        net_electrical_output = dispatch_soln.net_electrical_output,
                        day_ahead_schedule_time = self.day_ahead_schedule_time
                        )

                    weather_at_day_ahead_schedule = dispatch.get_weather_at_day_ahead_schedule(self.weather_data_for_dispatch, startpt, npts_horizon)
                    self.weather_at_schedule.append(weather_at_day_ahead_schedule)  # Store weather used at the point in time the day ahead schedule was generated

                #--- Set ssc dispatch targets
                ssc_dispatch_targets = dispatch.DispatchTargets(dispatch_soln, self.plant, self.dispatch_params, sscstep, freq/3600.)

                if start_date == datetime.datetime(2018, 10, 14, 0, 0):
                    assert hash(tuple(ssc_dispatch_targets.is_pc_sb_allowed_in)) == -4965923453060612375
                    assert hash(tuple(ssc_dispatch_targets.is_pc_su_allowed_in)) == -4965923453060612375
                    assert hash(tuple(ssc_dispatch_targets.is_rec_sb_allowed_in)) == -4965923453060612375
                    assert hash(tuple(ssc_dispatch_targets.is_rec_su_allowed_in)) == -4965923453060612375
                    assert hash(tuple(ssc_dispatch_targets.q_pc_max_in)) == -709626543671595165
                    assert hash(tuple(ssc_dispatch_targets.q_pc_target_on_in)) == -4965923453060612375
                    assert hash(tuple(ssc_dispatch_targets.q_pc_target_su_in)) == -4965923453060612375

                if start_date == datetime.datetime(2018, 10, 14, 1, 0):
                    assert hash(tuple(ssc_dispatch_targets.is_pc_sb_allowed_in)) == -4965923453060612375
                    assert hash(tuple(ssc_dispatch_targets.is_pc_su_allowed_in)) == -4965923453060612375
                    assert hash(tuple(ssc_dispatch_targets.is_rec_sb_allowed_in)) == -4965923453060612375
                    assert hash(tuple(ssc_dispatch_targets.is_rec_su_allowed_in)) == -4965923453060612375
                    assert hash(tuple(ssc_dispatch_targets.q_pc_max_in)) == -709626543671595165
                    assert hash(tuple(ssc_dispatch_targets.q_pc_target_on_in)) == -4965923453060612375
                    assert hash(tuple(ssc_dispatch_targets.q_pc_target_su_in)) == -4965923453060612375

                #--- Save these values for next estimates
                ursd_last = dispatch_soln.get_value_at_time(self.dispatch_params, freq/3600, 'ursd')      # set to False when it doesn't exists 
                yrsd_last = dispatch_soln.get_value_at_time(self.dispatch_params, freq/3600, 'yrsd')      # set to False when it doesn't exists

            else:  # Infeasible solution was returned, revert back to running ssc without dispatch targets
                pass
        

            #
            # Return ssc_dispatch_targets and Rdisp
            #

        
        ################################
        #--- Run ssc and collect results
        ################################
        # Start compiling ssc input dict (D2)
        D2 = self.plant.design.copy()
        D2.update(self.plant.state)
        D2.update(self.plant.flux_maps)
        D2['time_start'] = int(util.get_time_of_year(self.start_date))
        D2['time_stop'] = util.get_time_of_year(self.start_date.replace(hour=0, minute=0, second=0)) + self.sim_days*24*3600
        D2['sf_adjust:hourly'] = CaseStudy.get_field_availability_adjustment(self.ssc_time_steps_per_hour, self.start_date.year, self.control_field,
            self.use_CD_measured_reflectivity, self.plant.design, self.fixed_soiling_loss)
        CaseStudy.reupdate_ssc_constants(D2, self.params, self.data)
        if self.control_receiver == 'CD_data':
            CaseStudy.load_user_flow_paths(
                D=D2,
                flow_path_1_data=self.CD_mflow_path2_data,          # Note ssc path numbers are reversed relative to CD path numbers
                flow_path_2_data=self.CD_mflow_path1_data,
                use_measured_reflectivity=self.use_CD_measured_reflectivity,
                soiling_avail=util.get_CD_soiling_availability(self.start_date.year, D2['helio_reflectance'] * 100), # CD soiled / clean reflectivity (daily array),
                fixed_soiling_loss=self.fixed_soiling_loss
                )

        if self.is_optimize and dispatch_soln is not None:
            D2.update(vars(ssc_dispatch_targets))

        D2['time_stop'] = toy+freq
        R, new_plant_state = ssc_wrapper.call_ssc(D2, retvars, plant_state_pt = napply-1, npts = napply)
        
        #--- Update saved plant state
        persistance_vars = plant_.Plant.update_persistence(
            self.plant.state,
            R,
            new_plant_state['rec_op_mode_initial'],
            new_plant_state['pc_op_mode_initial'],
            sscstep/3600.)
        new_plant_state.update(persistance_vars)
        self.plant.state.update(new_plant_state)

        if start_date == datetime.datetime(2018, 10, 14, 0, 0):
            assert math.isclose(self.plant.state['pc_startup_energy_remain_initial'], 29339.9, rel_tol=1e-4)
            assert math.isclose(self.plant.state['pc_startup_time_remain_init'], 0.5, rel_tol=1e-4)
            assert math.isclose(self.plant.state['rec_startup_energy_remain_init'], 141250000, rel_tol=1e-4)
            assert math.isclose(self.plant.state['rec_startup_time_remain_init'], 1.15, rel_tol=1e-4)
            assert math.isclose(self.plant.state['disp_rec_persist0'], 1001, rel_tol=1e-4)
            assert math.isclose(self.plant.state['disp_rec_off0'], 1001, rel_tol=1e-4)
            assert math.isclose(self.plant.state['disp_pc_persist0'], 1001, rel_tol=1e-4)
            assert math.isclose(self.plant.state['disp_pc_off0'], 1001, rel_tol=1e-4)

        if start_date == datetime.datetime(2018, 10, 14, 1, 0):
            assert math.isclose(self.plant.state['pc_startup_energy_remain_initial'], 29339.9, rel_tol=1e-4)
            assert math.isclose(self.plant.state['pc_startup_time_remain_init'], 0.5, rel_tol=1e-4)
            assert math.isclose(self.plant.state['rec_startup_energy_remain_init'], 141250000, rel_tol=1e-4)
            assert math.isclose(self.plant.state['rec_startup_time_remain_init'], 1.15, rel_tol=1e-4)
            assert math.isclose(self.plant.state['disp_rec_persist0'], 1002, rel_tol=1e-4)
            assert math.isclose(self.plant.state['disp_rec_off0'], 1002, rel_tol=1e-4)
            assert math.isclose(self.plant.state['disp_pc_persist0'], 1002, rel_tol=1e-4)
            assert math.isclose(self.plant.state['disp_pc_off0'], 1002, rel_tol=1e-4)
        
        R.update(Rdisp)         # add in dispatch results
        self.results = R
        

        # TODO: Just remove this?
        # Read NVE schedules (if not already read during rolling horizon calculations)
        if self.is_optimize == False and self.use_day_ahead_schedule and self.day_ahead_schedule_from == 'NVE':
            for j in range(self.sim_days):
                date = datetime.datetime(self.start_date.year, self.start_date.month, self.start_date.day + j)
                self.schedules.append(self.get_CD_NVE_day_ahead_schedule(date))

        # Calculate post-simulation financials
        self.revenue = CaseStudy.calculate_revenue(self.start_date, self.sim_days, self.results['P_out_net'], self.params, self.data)
        day_ahead_penalties = CaseStudy.calculate_day_ahead_penalty(self.sim_days, self.schedules, self.results['P_out_net'], self.params, self.disp_soln_tracking, 
            self.disp_params_tracking)
        startup_ramping_penalties = self.calculate_startup_ramping_penalty()        
        
        outputs = {
            'revenue': self.revenue,
            'Q_thermal': self.results['Q_thermal'],
            'P_cycle': self.results['P_cycle'],
            'P_out_net': self.results['P_out_net']
        }
        outputs.update(day_ahead_penalties)
        outputs.update(startup_ramping_penalties)


        results = {
            'outputs': outputs,
            'plant_state': self.plant.state,
            'ursd_last': ursd_last,
            'yrsd_last': yrsd_last,
            'current_forecast_weather_data': self.current_forecast_weather_data,
	        'weather_data_for_dispatch': self.weather_data_for_dispatch,
            'schedules': self.schedules,
	        'current_day_schedule': self.current_day_schedule,
	        'next_day_schedule': self.next_day_schedule
        }

        return results
            

    def initialize(self):

        def adjust_plant_design(plant_design, cycle_type, user_defined_cycle_input_file):
            """Set cycle specifications (from model validation code)"""
            if cycle_type == 'user_defined':
                plant_design['P_ref'] = 120
                plant_design['design_eff'] = 0.409
                plant_design['T_htf_cold_des'] = 295.0 # [C]      # This sets design mass flowrate to that in CD's data
                plant_design['pc_config'] = 1
                with open(os.path.join(os.path.dirname(__file__), user_defined_cycle_input_file), 'r') as read_obj:
                    csv_reader = reader(read_obj)
                    plant_design['ud_ind_od'] = list(csv_reader)        
                for i in range(len(plant_design['ud_ind_od'])):
                    plant_design['ud_ind_od'][i] = [float(item) for item in plant_design['ud_ind_od'][i]]
                    
            elif cycle_type == 'sliding':  
                ### For sliding pressure
                ## These parameters work with heat input calculated using 290 as the lower temperature - however, there are a couple of controller issues
                plant_design['P_ref'] = 125
                plant_design['design_eff'] = 0.378
                plant_design['tech_type'] = 3
            
            else:
                ### For fixed pressure
                plant_design['P_ref'] = 120.
                plant_design['design_eff'] = 0.409  # 0.385
                plant_design['tech_type'] = 1
            
            return
        
        # Set cycle specifications (from model validation code)
        # TODO: do we actually want to do this? I would assume not.
        adjust_plant_design(self.plant.design, self.cycle_type, self.user_defined_cycle_input_file)
        
        # Check combinations of control conditions
        if self.is_optimize and (self.control_field == 'CD_data' or self.control_receiver == 'CD_data'):
            print ('Warning: Dispatch optimization is being used with field or receiver operation derived from CD data. Receiver can only operate when original CD receiver was operating')
        if self.control_receiver == 'CD_data' and self.control_field != 'CD_data':
            print ('Warning: Receiver flow is controlled from CD data, but field tracking fraction is controlled by ssc. Temperatures will likely be unrealistically high')

        self.is_initialized = True
        return
    
    @staticmethod
    def update_forecast_weather_data(date, current_forecast_weather_data, ssc_time_steps_per_hour, forecast_steps_per_hour, ground_truth_weather_data,
                                     forecast_issue_time, day_ahead_schedule_time, clearsky_data):
        """
        Inputs:
            date
            current_forecast_weather_data
            ssc_time_steps_per_hour
            forecast_steps_per_hour
            ground_truth_weather_data
            forecast_issue_time
            day_ahead_schedule_time
            clearsky_data

        Outputs:
            current_forecast_weather_data
        """

        offset30 = True
        print ('Updating weather forecast:', date)
        nextdate = date + datetime.timedelta(days = 1) # Forecasts issued at 4pm PST on a given day (PST) are labeled at midnight (UTC) on the next day 
        wfdata = util.read_weather_forecast(nextdate, offset30)
        t = int(util.get_time_of_year(date)/3600)   # Time of year (hr)
        pssc = int(t*ssc_time_steps_per_hour) 
        nssc_per_wf = int(ssc_time_steps_per_hour / forecast_steps_per_hour)
        
        #---Update forecast data in full weather file: Assuming forecast points are on half-hour time points, valid for the surrounding hour, with the first point 30min prior to the designated forecast issue time
        if not offset30:  # Assume forecast points are on the hour, valid for the surrounding hour
            n = len(wfdata['dn'])
            for j in range(n): # Time points in weather forecast
                q  = pssc + nssc_per_wf/2  if j == 0 else pssc + nssc_per_wf/2 + (j-1)*nssc_per_wf/2  # First point in annual weather data (at ssc time resolution) for forecast time point j
                nuse = nssc_per_wf/2 if j==0 else nssc_per_wf 
                for k in ['dn', 'wspd', 'tdry', 'rhum', 'pres']:
                    val =  wfdata[k][j] if k in wfdata.keys() else ground_truth_weather_data[k][pssc]  # Use current ground-truth value for full forecast period if forecast value is not available            
                    for i in range(nuse):  
                        current_forecast_weather_data[k][q+i] = val   
                
        else: # Assume forecast points are on the half-hour, valid for the surrounding hour, with the first point 30min prior to the designated forecast issue time
            n = len(wfdata['dn']) - 1
            for j in range(n): 
                q = pssc + j*nssc_per_wf
                for k in ['dn', 'wspd', 'tdry', 'rhum', 'pres']:
                    val =  wfdata[k][j+1] if k in wfdata.keys() else ground_truth_weather_data[k][pssc]  # Use current ground-truth value for full forecast period if forecast value is not available            
                    for i in range(nssc_per_wf):  
                        current_forecast_weather_data[k][q+i] = val

        #--- Extrapolate forecasts to be complete for next-day dispatch scheduling (if necessary)
        forecast_duration = n*forecast_steps_per_hour if offset30 else (n-0.5)*forecast_steps_per_hour
        if forecast_issue_time > day_ahead_schedule_time:
            hours_avail = forecast_duration - (24 - forecast_issue_time) - day_ahead_schedule_time  # Hours of forecast available at the point the day ahead schedule is due
        else:
            hours_avail = forecast_duration - (day_ahead_schedule_time - forecast_issue_time)
            
        req_hours_avail = 48 - day_ahead_schedule_time 
        if req_hours_avail >  hours_avail:  # Forecast is not available for the full time required for the day-ahead schedule
            qf = pssc + int((n-0.5)*nssc_per_wf) if offset30 else pssc + (n-1)*nssc_per_wf   # Point in annual arrays corresponding to last point forecast time point
            cratio = 0.0 if wfdata['dn'][-1]<20 else wfdata['dn'][-1] / max(clearsky_data[qf], 1.e-6)  # Ratio of actual / clearsky at last forecast time point
            
            nmiss = int((req_hours_avail - hours_avail) * ssc_time_steps_per_hour)  
            q = pssc + n*nssc_per_wf if offset30 else pssc + int((n-0.5)*nssc_per_wf ) 
            for i in range(nmiss):
                current_forecast_weather_data['dn'][q+i] = clearsky_data[q+i] * cratio    # Approximate DNI in non-forecasted time periods from expected clear-sky DNI and actual/clear-sky ratio at latest available forecast time point
                for k in ['wspd', 'tdry', 'rhum', 'pres']:  
                    current_forecast_weather_data[k][q+i] = current_forecast_weather_data[k][q-1]  # Assume latest forecast value applies for the remainder of the time period

        return current_forecast_weather_data
    
    
    @staticmethod
    def get_field_availability_adjustment(steps_per_hour, year, control_field, use_CD_measured_reflectivity, plant_design, fixed_soiling_loss):
        """
        Inputs:
            steps_per_hour
            year
            control_field
            use_CD_measured_reflectivity
            plant.design
                N_hel
                helio_reflectance
            fixed_soiling_loss

        Outputs:
            adjust
        """

        if control_field == 'ssc':
            if use_CD_measured_reflectivity:
                adjust = util.get_field_adjustment_from_CD_data(year, plant_design['N_hel'], plant_design['helio_reflectance']*100, True, None, False)            
            else:
                adjust = (fixed_soiling_loss * 100 * np.ones(steps_per_hour*24*365))  

        elif control_field == 'CD_data':
            if use_CD_measured_reflectivity:
                adjust = util.get_field_adjustment_from_CD_data(year, plant_design['N_hel'], plant_design['helio_reflectance']*100, True, None, True)
            else:
                refl = (1-fixed_soiling_loss) * plant_design['helio_reflectance'] * 100  # Simulated heliostat reflectivity
                adjust = util.get_field_adjustment_from_CD_data(year, plant_design['N_hel'], plant_design['helio_reflectance']*100, False, refl, True)
 
        adjust = adjust.tolist()
        data_steps_per_hour = len(adjust)/8760  
        if data_steps_per_hour != steps_per_hour:
            adjust = util.translate_to_new_timestep(adjust, 1./data_steps_per_hour, 1./steps_per_hour)
        return adjust
    

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


    # Calculate revenue
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

    
    def calculate_startup_ramping_penalty(self):
        """
        Inputs:
            q_startup
            Q_thermal
            q_pb
            P_cycle
            q_dot_pc_startup
            Crsu                    plant.design
            Ccsu                    plant.design
            C_delta_w               plant.design

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

        self.n_starts_rec, self.n_starts_rec_attempted = find_starts(self.results['q_startup'], self.results['Q_thermal'])

        qpb_on = self.results['q_pb']   # Cycle thermal power includes startup
        inds_off = np.where(self.results['P_cycle']<1.e-3)[0]
        qpb_on[inds_off] = 0.0
        self.n_starts_cycle, self.n_starts_cycle_attempted = find_starts(self.results['q_dot_pc_startup'], qpb_on)

        n = len(self.results['P_cycle'])
        self.cycle_ramp_up = 0.0
        self.cycle_ramp_down = 0.0
        w = self.results['P_cycle']
        for j in range(1,n):
            diff =  w[j] - w[j-1]
            if diff > 0:
                self.cycle_ramp_up += diff
            elif diff < 0:
                self.cycle_ramp_down += (-diff)
        
        self.startup_ramping_penalty = self.n_starts_rec*self.plant.design['Crsu'] + self.n_starts_cycle*self.plant.design['Ccsu'] 
        self.startup_ramping_penalty += self.cycle_ramp_up*self.plant.design['C_delta_w']*1000 + self.cycle_ramp_down*self.plant.design['C_delta_w']*1000

        outputs = {
            'n_starts_rec': self.n_starts_rec,
            'n_starts_rec_attempted': self.n_starts_rec_attempted,
            'n_starts_cycle': self.n_starts_cycle,
            'n_starts_cycle_attempted': self.n_starts_cycle_attempted,
            'cycle_ramp_up': self.cycle_ramp_up,
            'cycle_ramp_down': self.cycle_ramp_down,
            'startup_ramping_penalty': self.startup_ramping_penalty
        }
        return outputs
        
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
    

    def default_disp_stored_vars(self):
        return ['cycle_on', 'cycle_standby', 'cycle_startup', 'receiver_on', 'receiver_startup', 'receiver_standby', 
               'receiver_power', 'thermal_input_to_cycle', 'electrical_output_from_cycle', 'net_electrical_output', 'tes_soc',
               'yrsd', 'ursd']
    

    @staticmethod
    def load_user_flow_paths(D, flow_path_1_data, flow_path_2_data, use_measured_reflectivity, soiling_avail=None, fixed_soiling_loss=None):
        """Load user flow paths into the ssc input dict (D)"""
        mult = np.ones_like(flow_path_1_data)
        if not use_measured_reflectivity:                                                   # Scale mass flow based on simulated reflectivity vs. CD actual reflectivity
            rho = (1-fixed_soiling_loss) * D['helio_reflectance']                           # Simulated heliostat reflectivity
            mult = rho*np.ones(365) / (soiling_avail*D['helio_reflectance'])                # Ratio of simulated / CD reflectivity
            mult = np.repeat(mult, 24*D['time_steps_per_hour'])                             # Annual array at ssc resolution
        D['rec_user_mflow_path_1'] = (flow_path_1_data * mult).tolist()                     # Note ssc path numbers are reversed relative to CD path numbers
        D['rec_user_mflow_path_2'] = (flow_path_2_data * mult).tolist()





########################################################################################################################################################
if __name__ == '__main__':
    start = timeit.default_timer()
    os.chdir(os.path.dirname(__file__))
    
    # Mediator parameters
    m_vars = mediator_params.copy()
    ssc_time_steps_per_hour = m_vars['time_steps_per_hour']


    # Data - get historical weather
    ground_truth_weather_file = './model-validation/input_files/weather_files/ssc_weatherfile_1min_2018.csv'  # Weather file derived from CD data: DNI, ambient temperature,
                                                                                                                   #  wind speed, etc. are averaged over 4 CD weather stations,
                                                                                                                   #  after filtering DNI readings for bad measurements. 
    ground_truth_weather_data = util.read_weather_data(ground_truth_weather_file)
    if ssc_time_steps_per_hour != 60:
        ground_truth_weather_data = util.update_weather_timestep(ground_truth_weather_data, ssc_time_steps_per_hour)

    # Data - get clear-sky DNI annual arrays
    clearsky_file = './model-validation/input_files/weather_files/clearsky_pvlib_ineichen_1min_2018.csv'      # Expected clear-sky DNI from Ineichen model (via pvlib).  
    clearsky_data = np.genfromtxt(clearsky_file)
    if ssc_time_steps_per_hour != 60:
        clearsky_data = np.array(util.translate_to_new_timestep(clearsky_data, 1./60, 1./ssc_time_steps_per_hour))

    # Data - get receiver mass flow annual arrays
    CD_mflow_path1_file = './model-validation/input_files/mflow_path1_2018_1min.csv'                          # File containing CD data for receiver path 1 mass flow rate (note, all values are zeros on days without data)
    CD_mflow_path2_file = './model-validation/input_files/mflow_path2_2018_1min.csv'                          # File containing CD data for receiver path 2 mass flow rate (note, all values are zeros on days without data)
    CD_mflow_path1_data = np.genfromtxt(CD_mflow_path1_file)
    CD_mflow_path2_data = np.genfromtxt(CD_mflow_path2_file)
    if ssc_time_steps_per_hour != 60:
        CD_mflow_path1_data = np.array(util.translate_to_new_timestep(CD_mflow_path1_data, 1./60, 1./ssc_time_steps_per_hour))
        CD_mflow_path2_data = np.array(util.translate_to_new_timestep(CD_mflow_path2_data, 1./60, 1./ssc_time_steps_per_hour))

    # Data - get prices
    price_multiplier_file = 'prices_flat.csv'  # TODO: File containing annual price multipliers
    price_multipliers = np.genfromtxt(price_multiplier_file)
    if m_vars['price_steps_per_hour'] != ssc_time_steps_per_hour:
        price_multipliers = util.translate_to_new_timestep(price_multipliers, 1./m_vars['price_steps_per_hour'], 1./ssc_time_steps_per_hour)
    pmavg = sum(price_multipliers)/len(price_multipliers)  
    price_data = [m_vars['avg_price']*p/pmavg  for p in price_multipliers]  # Electricity price at ssc time steps ($/MWh)

    data = {
        'dispatch_factors_ts':              price_data,
        'clearsky_data':                    clearsky_data,
        'solar_resource_data':              ground_truth_weather_data,
        'CD_mflow_path1_data':              CD_mflow_path1_data,
        'CD_mflow_path2_data':              CD_mflow_path2_data,
    }

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


    # Dispatch inputs
    start_date = datetime.datetime(2018, 10, 14)
    sim_days = 1
    horizon = 86400                 # TODO  make this not hardcoded
    ursd_last = 0
    yrsd_last = 0  
    weather_data_for_dispatch = util.create_empty_weather_data(ground_truth_weather_data, ssc_time_steps_per_hour)
    current_day_schedule = np.zeros(24*m_vars['day_ahead_schedule_steps_per_hour'])
    next_day_schedule = np.zeros(24*m_vars['day_ahead_schedule_steps_per_hour'])
    CD_raw_data_direc = './input_files/CD_raw'                     # Directory containing raw data files from CD
    CD_processed_data_direc = './input_files/CD_processed'         # Directory containing files with 1min data already extracted
    initial_plant_state = util.get_initial_state_from_CD_data(start_date, CD_raw_data_direc, CD_processed_data_direc, plant.design)
    current_forecast_weather_data = CaseStudy.update_forecast_weather_data(
                date=start_date - datetime.timedelta(hours = 24-m_vars['forecast_issue_time']),
                current_forecast_weather_data=util.create_empty_weather_data(ground_truth_weather_data, ssc_time_steps_per_hour),
                ssc_time_steps_per_hour=ssc_time_steps_per_hour,
                forecast_steps_per_hour=m_vars['forecast_steps_per_hour'],
                ground_truth_weather_data=ground_truth_weather_data,
                forecast_issue_time=m_vars['forecast_issue_time'],
                day_ahead_schedule_time=m_vars['day_ahead_schedule_time'],
                clearsky_data=clearsky_data
                )
    schedules = []
    if int(util.get_time_of_day(start_date)) == 0 and m_vars['use_day_ahead_schedule'] and m_vars['day_ahead_schedule_from'] == 'calculated':
        schedules.append(None)


    # Run models
    outputs_total = {}
    nupdate = int(sim_days*24 / m_vars['dispatch_frequency'])
    for j in range(nupdate):
        tod = int(util.get_time_of_day(start_date))
        cs = CaseStudy(plant=plant, params=m_vars, data=data)
        results = cs.run(
            start_date=start_date,
            timestep_days=m_vars['dispatch_frequency']/24.,
            horizon=horizon,
            ursd_last=ursd_last,
            yrsd_last=yrsd_last,
            current_forecast_weather_data=current_forecast_weather_data,
            weather_data_for_dispatch=weather_data_for_dispatch,
            schedules=schedules,
            current_day_schedule=current_day_schedule,
            next_day_schedule=next_day_schedule,
            initial_plant_state=initial_plant_state
            )

        # Update inputs for next call
        schedules = results['schedules']
        current_day_schedule = results['current_day_schedule']
        next_day_schedule = results['next_day_schedule']
        if tod == 0 and m_vars['use_day_ahead_schedule'] and m_vars['day_ahead_schedule_from'] == 'calculated':
            current_day_schedule = [s for s in next_day_schedule]
            schedules.append(current_day_schedule)
        start_date += datetime.timedelta(hours=m_vars['dispatch_frequency'])
        horizon -= 3600         # TODO make this not hardcoded
        ursd_last = results['ursd_last']
        yrsd_last = results['yrsd_last']
        current_forecast_weather_data = results['current_forecast_weather_data']
        weather_data_for_dispatch = results['weather_data_for_dispatch']
        initial_plant_state = results['plant_state']

        # Aggregate totals
        outputs = results['outputs']
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

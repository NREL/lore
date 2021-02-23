
import numpy as np
from csv import reader
from copy import deepcopy
import timeit
import plant_design
import dispatch
import ssc_wrapper
import util
import run_phase_one
import pandas

import datetime



class CaseStudy:
    def __init__(self, isdebug = False):

        #--- Plant design and operating properties
        self.design = plant_design.PlantDesign()           # Default parameters contain best representation of CD plant -> updated in "initialize" depending on specified cycle type
        self.properties = plant_design.PlantProperties()   # Default parameters contain best representation of CD plant and dispatch properties

        #--- Simulation start point and duration
        self.start_date = datetime.datetime(2018, 8, 31)   # Start date for simulations
        self.sim_days = 2                                  # Number of days to simulate
        self.set_initial_state_from_CD_data = True         # Set initial plant state based on CD data?
        
        
        #--- Control conditions
        self.is_optimize = True        # Use dispatch optimziation
        self.control_field = 'ssc'     #'CD_data' = use CD data to control heliostats tracking, heliostats offline, and heliostat daily reflectivity.  Receiver startup time is set to zero so that simulated receiver starts when CD tracking begins
                                       #'ssc' = allow ssc to control heliostat field operations, assuming all heliostats are available
                                       
        self.control_receiver = 'ssc_clearsky'    #'CD_data' = use CD data to control receiver mass flow. Receiver startup time is set to zero so that simulated receiver starts when CD receiver finishes startup
                                                  #'ssc_clearsky' = use expected clearsky DNI to control receiver mass flow.  If field_control = 'ssc' then the historical median startup time from CD data will be used to control receiver startup
                                                  #'ssc_actual_dni' = use actual DNI to control receiver mass flow.  If field_control = 'ssc' then the historical median startup time from CD data will be used to control receiver startup   

        self.control_cycle = 'ssc_heuristic'     # Only used if is_optimize = False
                                                 # 'CD_data' = use CD actual cycle operation to control cycle dispatch targets
                                                 # 'ssc_heuristic' = allow ssc heuristic (no consideration of TOD price) to control cycle dispatch


        #--- Time steps, time horizons, and update intervals
        self.ssc_time_steps_per_hour = 60               # Simulation time resolution in ssc (1min)

        # Dispatch optimization
        self.dispatch_horizon = 48.                     # Dispatch time horizon (hr) 
        self.dispatch_frequency = 1.0                   # Frequency of dispatch re-optimization (hr)
        self.dispatch_horizon_update = 24.              # Frequency of dispatch time horizon update (hr) -> set to the same value as dispatch_frequency for a fixed-length horizon 
        self.dispatch_steplength_array = [5, 15, 60]    # Dispatch time step sizes (min)
        self.dispatch_steplength_end_time = [1, 4, 48]  # End time for dispatch step lengths (hr)
        self.nonlinear_model_time = 4.0                 # Amount of time to apply nonlinear dispatch model (hr) (not currently used)
        self.disp_time_weighting = 0.999                # Dispatch time weighting factor. 
        self.dispatch_weather_horizon = -1              # Time point in hours (relative to start of optimization horizon) defining the transition from actual weather to forecasted weather used in the dispatch model. Set to 0 to use forecasts for the full horizon, set to -1 to use actual weather for the full horizon, or any value > 0 to combine actual/forecasted weather                                              
        self.use_linear_dispatch_at_night = False       # Revert to the linear dispatch model when all of the time-horizon in the nonlinear model is at night.
        self.night_clearsky_cutoff = 100.               # Cutoff value for clear-sky DNI defining "night"

        # Weather forecasts                                                
        self.forecast_steps_per_hour = 1                # Number of time steps per hour in weather forecasts
        self.forecast_update_frequency = 24             # Weather forecast update interval (hr)
        self.forecast_issue_time = 16                   # Time at which weather forecast is issued (hr, 0-23), assumed to be in standard time.  Forecasts issued at midnight UTC 

        # Price
        self.price_steps_per_hour = 1                   # Number of steps per hour in electricity price multipliers
        self.avg_price = 138                            # Average electricity price ($/MWh):  CD original PPA was $138/MWh
        self.avg_purchase_price = 30                    # Average electricity purchase price ($/MWh) (note, this isn't currently used in the dispatch model)
        self.avg_price_disp_storage_incentive = 0.0     # Average electricity price ($/MWh) used in dispatch model storage inventory incentive

        # Day-ahead schedule targets
        self.use_day_ahead_schedule = True              # Use day-ahead generation targets
        self.day_ahead_schedule_from = 'calculated'     # 'calculated' = calculate day-ahead schedule during solution, 'NVE'= use NVE-provided schedule for CD
        self.day_ahead_schedule_time = 10               # Time of day at which day-ahead schedule is due (hr, 0-23), assumed to be in standard time
        self.day_ahead_schedule_steps_per_hour = 1      # Time resolution of day-ahead schedule
        self.day_ahead_pen_plus = 500                   # Penalty for over-generation relative to day-ahead schedule ($/MWhe)
        self.day_ahead_pen_minus = 500                  # Penalty for under-generation relative to day-ahead schedule ($/MWhe)
        self.day_ahead_tol_plus = 5                     # Tolerance for over-production relative to day-ahead schedule before incurring penalty (MWhe)
        self.day_ahead_tol_minus = 5                    # Tolerance for under-production relative to day-ahead schedule before incurring penalty (MWhe)
        self.day_ahead_ignore_off = True                # Don't apply schedule penalties when cycle is scheduled to be off for the full hour (MWhe)

        
        #--- Field, receiver, and cycle simulation options
        self.use_CD_measured_reflectivity = True  # Use measured heliostat reflectivity from CD data
        self.fixed_soiling_loss = 0.035            # Fixed soiling loss (if not using CD measured reflectivity) = 1 - (reflectivity / clean_reflectivity)
        self.use_transient_startup = False  # TODO: Disabling transient startup -> ssc not yet configured to start/stop calculations in the middle of startup with this model
        self.use_transient_model = False    # TODO: Disabling transient receiver model -> ssc not yet configured to store/retrieve receiver temperature profiles
        self.cycle_type = 'user_defined'   # 'user-defined', 'sliding', or 'fixed'



        #--- Input data files: weather, masslow, clearsky DNI must have length of full annual array based on ssc time step size
        self.user_defined_cycle_input_file = 'udpc_noTamb_dependency.csv'  # Only required if cycle_type is user_defined
        self.price_multiplier_file = 'prices_flat.csv'   # TODO: File containing annual price multipliers
        self.ground_truth_weather_file = '../model-validation/input_files/weather_files/ssc_weatherfile_1min_2018.csv'  # Weather file derived from CD data: DNI, ambient temperature, wind speed, etc. are averaged over 4 CD weather stations, after filtering DNI readings for bad measurements. 
        self.clearsky_file = '../model-validation/input_files/weather_files/clearsky_pvlib_ineichen_1min_2018.csv'      # Expected clear-sky DNI from Ineichen model (via pvlib).  
        self.CD_mflow_path1_file = '../model-validation/input_files/mflow_path1_2018_1min.csv'                          # File containing CD data for receiver path 1 mass flow rate (note, all values are zeros on days without data)
        self.CD_mflow_path2_file = '../model-validation/input_files/mflow_path2_2018_1min.csv'                          # File containing CD data for receiver path 2 mass flow rate (note, all values are zeros on days without data)
        
        self.CD_raw_data_direc = '../../../Crescent Dunes data/NREL - CD collaboration/Steam Generation/Daily Reports/'  # Directory containing raw data files from CD
        self.CD_processed_data_direc = '../../../Crescent Dunes data/Daily operations data/'                             # Directory containing files with 1min data already extracted

        self.CD_data_for_plotting = {}  # Only used if control_cycle = 'CD_data' 

        #--- Miscellaneous
        self.store_full_dispatch_solns = True           # Store full dispatch input parameters and solutions for each call to the dispatch model
        self.force_rolling_horizon = False              # Force simulation using ssc heuristic dispatch to follow a rolling horizon for debugging?
        self.is_debug = isdebug                         # Reduce resolution in flux profile for faster solution
        self.save_results_to_file = False               # Save results to file
        self.results_file = 'case_study'                # Filename to save results: 
        
        
        
        #--- Calculated values
        self.plant_state = ssc_wrapper.PlantState()                 # Structure to contain current plant state
        self.flux_maps = ssc_wrapper.FluxMaps()                     # Structure to contain computed flux maps
        self.dispatch_params = dispatch.DispatchParams()            # Structure to contain all inputs for dispatch model 
        self.dispatch_soln = dispatch.DispatchSoln()                # Structure to contain current dispatch optimization solution
        self.ssc_dispatch_targets = ssc_wrapper.DispatchTargets()   # Structure to contain dispatch targets used for ssc

        self.current_time = 0                           # Current time (tracked in standard time, not local time)
        self.is_initialized = False                     # Has solution already been initalized?        
        self.results = None                             # Results

        self.current_day_schedule = []                  # Committed day-ahead generation schedule for current day (MWe)
        self.next_day_schedule = []                     # Predicted day-ahead generation schedule for next day (MWe)
        self.schedules = []                             # List to store all day-ahead generation schedules (MWe)
        self.weather_at_schedule = []                   # List to store weather data at the point in time the day-ahead schedule was generated
        self.disp_params_tracking = []                  # List to store dispatch parameters for each call to dispatch model
        self.disp_soln_tracking = []                    # List to store dispatch solutions for each call to dispatch model (directly from dispatch model, no translation to ssc time steps)
        self.plant_state_tracking = []                  # List to store plant state at the start of each call to the dispatch model
        self.infeasible_count = 0
        
        self.revenue = 0.0                      # Revenue over simulated days ($)
        self.startup_ramping_penalty = 0.0      # Startup and ramping penalty over all simulated days ($)
        self.day_ahead_penalty_tot = {}         # Penalty for missing schedule over all simulated days ($)
        self.day_ahead_diff_tot = {}            # Total difference between actual and scheduled generation (MWhe)
        self.day_ahead_diff_over_tol_plus = {}  # Total (positive) difference between actual and scheduled generation over tolerance (MWhe)
        self.day_ahead_diff_over_tol_minus = {} # Total (negative) difference between actual and scheduled generation over tolerance (MWhe)
        
        self.day_ahead_diff = []                 # Difference between net generation and scheduled net generation (MWhe)
        self.day_ahead_penalty = []              # Penalty for difference between net generation and scheduled net generation (MWhe)
        self.day_ahead_diff_ssc_disp_gross = []  # Difference between ssc and dispatch gross generation in schedule time steps (MWhe)
        
        self.n_starts_rec = 0               # Number of receiver starts
        self.n_starts_rec_attempted = 0     # Number of receiver starts, including those not completed
        self.n_starts_cycle = 0             # Number of cycle starts
        self.n_starts_cycle_attempted = 0   # Number of cycle starts, including those not completed
        self.cycle_ramp_up = 0              # Cycle ramp-up (MWe)
        self.cycle_ramp_down = 0            # Cycle ramp-down (MWe)
        self.total_receiver_thermal = 0     # Total thermal energy from receiver (GWht)
        self.total_cycle_gross = 0          # Total gross generation by cycle (GWhe)
        self.total_cycle_net = 0            # Total net generation by cycle (GWhe)

        return


    #------------------------------------------------------------------------
    
    def initialize(self):
        
        # Set cycle specifications (from model validation code)
        if self.cycle_type == 'user_defined':
            self.design.P_ref = 120
            self.design.design_eff = 0.409
            self.design.T_htf_cold_des = 295.0 # [C]      # This sets design mass flowrate to that in CD's data
            self.design.pc_config = 1
            with open(self.user_defined_cycle_input_file, 'r') as read_obj:
                csv_reader = reader(read_obj)
                self.design.ud_ind_od = list(csv_reader)        
            for i in range(len(self.design.ud_ind_od)):
                self.design.ud_ind_od[i] = [float(item) for item in self.design.ud_ind_od[i]]
                
        elif self.cycle_type == 'sliding':  
            ### For sliding pressure
            ## These parameters work with heat input calculated using 290 as the lower temperature - however, there are a couple of controller issues
            self.design.P_ref = 125
            self.design.design_eff = 0.378
            self.design.tech_type = 3
        
        else:
            ### For fixed pressure
            self.design.P_ref = 120.
            self.design.design_eff = 0.409  # 0.385
            self.design.tech_type = 1
            
        
        # Check combinations of control conditions
        if self.is_optimize and (self.control_field == 'CD_data' or self.control_receiver == 'CD_data'):
            print ('Warning: Dispatch optimization is being used with field or receiver operation derived from CD data. Receiver can only operate when original CD receiver was operating')
        if self.control_receiver == 'CD_data' and self.control_field != 'CD_data':
            print ('Warning: Receiver flow is controlled from CD data, but field tracking fraction is controlled by ssc. Temperatures will likely be unrealistically high')
        
        # Initialize design parameters
        self.design.initialize()

        # Read in historical weather data
        self.ground_truth_weather_data = util.read_weather_data(self.ground_truth_weather_file)
        if self.ssc_time_steps_per_hour != 60:
            self.ground_truth_weather_data = util.update_weather_timestep(self.ground_truth_weather_data, self.ssc_time_steps_per_hour)
        

        # Read in annual arrays for clear-sky DNI, receiver mass flow, etc.
        self.clearsky_data = np.genfromtxt(self.clearsky_file)
        self.CD_mflow_path1_data = np.genfromtxt(self.CD_mflow_path1_file)
        self.CD_mflow_path2_data = np.genfromtxt(self.CD_mflow_path2_file)
        if self.ssc_time_steps_per_hour != 60:
            self.clearsky_data = np.array(util.translate_to_new_timestep(self.clearsky_data, 1./60, 1./self.ssc_time_steps_per_hour))
            self.CD_mflow_path1_data = np.array(util.translate_to_new_timestep(self.CD_mflow_path1_data, 1./60, 1./self.ssc_time_steps_per_hour))
            self.CD_mflow_path2_data = np.array(util.translate_to_new_timestep(self.CD_mflow_path2_data, 1./60, 1./self.ssc_time_steps_per_hour))            
        
        
        
        price_multipliers = np.genfromtxt(self.price_multiplier_file)
        if self.price_steps_per_hour != self.ssc_time_steps_per_hour:
            price_multipliers = util.translate_to_new_timestep(price_multipliers, 1./self.price_steps_per_hour, 1./self.ssc_time_steps_per_hour)
        pmavg = sum(price_multipliers)/len(price_multipliers)  
        self.price_data = [self.avg_price*p/pmavg  for p in price_multipliers]  # Electricity price at ssc time steps ($/MWh)

        # Create annual weather data structure that will contain weather forecast data to be used during optimization (weather data filled with all zeros for now)
        self.current_forecast_weather_data = util.create_empty_weather_data(self.ground_truth_weather_data, self.ssc_time_steps_per_hour)
        self.weather_data_for_dispatch = util.create_empty_weather_data(self.ground_truth_weather_data, self.ssc_time_steps_per_hour)

        # Initialize forecast weather data using the day prior to the first simulated day
        forecast_time = self.start_date - datetime.timedelta(hours = 24-self.forecast_issue_time)
        self.update_forecast_weather_data(forecast_time)

        # Initalize plant state
        self.plant_state.set_default(self.design, self.properties)
        if self.set_initial_state_from_CD_data:
            initial_state = util.get_initial_state_from_CD_data(self.start_date, self.CD_raw_data_direc, self.CD_processed_data_direc, self.design, self.properties)
            if initial_state is not None:
                self.plant_state = initial_state

        
        # Initialize day-ahead generation schedules
        n = 24*self.day_ahead_schedule_steps_per_hour
        self.current_day_schedule = np.zeros(n)
        self.next_day_schedule = np.zeros(n)

        self.is_initialized = True
        return



    # Update forecasted weather data 
    def update_forecast_weather_data(self, date, offset30 = True):
        print ('Updating weather forecast:', date)
        nextdate = date + datetime.timedelta(days = 1) # Forecasts issued at 4pm PST on a given day (PST) are labeled at midnight (UTC) on the next day 
        wfdata = util.read_weather_forecast(nextdate, offset30)
        t = int(util.get_time_of_year(date)/3600)   # Time of year (hr)
        pssc = int(t*self.ssc_time_steps_per_hour) 
        nssc_per_wf = int(self.ssc_time_steps_per_hour / self.forecast_steps_per_hour)
        
        #---Update forecast data in full weather file: Assuming forecast points are on half-hour time points, valid for the surrounding hour, with the first point 30min prior to the designated forecast issue time
        if not offset30:  # Assume forecast points are on the hour, valid for the surrounding hour
            n = len(wfdata['dn'])
            for j in range(n): # Time points in weather forecast
                q  = pssc + nssc_per_wf/2  if j == 0 else pssc + nssc_per_wf/2 + (j-1)*nssc_per_wf/2  # First point in annual weather data (at ssc time resolution) for forecast time point j
                nuse = nssc_per_wf/2 if j==0 else nssc_per_wf 
                for k in ['dn', 'wspd', 'tdry', 'rhum', 'pres']:
                    val =  wfdata[k][j] if k in wfdata.keys() else self.ground_truth_weather_data[k][pssc]  # Use current ground-truth value for full forecast period if forecast value is not available            
                    for i in range(nuse):  
                        self.current_forecast_weather_data[k][q+i] = val   
                
        else: # Assume forecast points are on the half-hour, valid for the surrounding hour, with the first point 30min prior to the designated forecast issue time
            n = len(wfdata['dn']) - 1
            for j in range(n): 
                q = pssc + j*nssc_per_wf
                for k in ['dn', 'wspd', 'tdry', 'rhum', 'pres']:
                    val =  wfdata[k][j+1] if k in wfdata.keys() else self.ground_truth_weather_data[k][pssc]  # Use current ground-truth value for full forecast period if forecast value is not available            
                    for i in range(nssc_per_wf):  
                        self.current_forecast_weather_data[k][q+i] = val

        #--- Extrapolate forecasts to be complete for next-day dispatch scheduling (if necessary)
        forecast_duration = n*self.forecast_steps_per_hour if offset30 else (n-0.5)*self.forecast_steps_per_hour
        if self.forecast_issue_time > self.day_ahead_schedule_time:
            hours_avail = forecast_duration - (24 - self.forecast_issue_time) - self.day_ahead_schedule_time  # Hours of forecast available at the point the day ahead schedule is due
        else:
            hours_avail = forecast_duration - (self.day_ahead_schedule_time - self.forecast_issue_time)
            
        req_hours_avail = 48 - self.day_ahead_schedule_time 
        if req_hours_avail >  hours_avail:  # Forecast is not available for the full time required for the day-ahead schedule
            qf = pssc + int((n-0.5)*nssc_per_wf) if offset30 else pssc + (n-1)*nssc_per_wf   # Point in annual arrays corresponding to last point forecast time point
            cratio = 0.0 if wfdata['dn'][-1]<20 else wfdata['dn'][-1] / max(self.clearsky_data[qf], 1.e-6)  # Ratio of actual / clearsky at last forecast time point
            
            nmiss = int((req_hours_avail - hours_avail) * self.ssc_time_steps_per_hour)  
            q = pssc + n*nssc_per_wf if offset30 else pssc + int((n-0.5)*nssc_per_wf ) 
            for i in range(nmiss):
                self.current_forecast_weather_data['dn'][q+i] = self.clearsky_data[q+i] * cratio    # Approximate DNI in non-forecasted time periods from expected clear-sky DNI and actual/clear-sky ratio at latest available forecast time point
                for k in ['wspd', 'tdry', 'rhum', 'pres']:  
                    self.current_forecast_weather_data[k][q+i] = self.current_forecast_weather_data[k][q-1]  # Assume latest forecast value applies for the remainder of the time period

        return
    
    
    # Update dispatch weather data 
    def update_dispatch_weather_data(self, weather_data, replacement_real_weather_data, replacement_forecast_weather_data, datetime, total_horizon, dispatch_horizon):
        """
        Replace select metrics in weather_data with those from the real and forecast weather data, depending on dispatch horizons
        """
        minutes_per_timestep = weather_data['minute'][1] - weather_data['minute'][0]
        timesteps_per_hour = 1 / minutes_per_timestep * 60

        t = int(util.get_time_of_year(datetime)/3600)        # Time of year (hr)
        p = int(t*timesteps_per_hour)                        # First time index in weather arrays
        n = int(total_horizon * timesteps_per_hour)                # Number of time indices in horizon (to replace)
        for j in range(n):
            for k in ['dn', 'wspd', 'tdry', 'rhum', 'pres']:
                hr = j/timesteps_per_hour
                if dispatch_horizon == -1 or hr < dispatch_horizon:
                    weather_data[k][p+j] = replacement_real_weather_data[k][p+j]    
                else:
                    weather_data[k][p+j] = replacement_forecast_weather_data[k][p+j]

        return weather_data
    
    def get_dispatch_targets_from_CD_actuals(self, use_avg_flow = False, set_rec_sb = False, ctrl_adj = False):
        # initialize targets stucture
        targets = ssc_wrapper.DispatchTargets()

        CD_plot_data = ['Gross Power [MW]', 'Net Power [MW]', 'E charge TES [MWht]', 'Hot Tank Temp [C]', 'Cold Tank Temp [C]', 'Rec avg Tout [C]']
        for key in CD_plot_data:
            self.CD_data_for_plotting[key] = []

        pc_on = []
        need_last_hour = False
        date = self.start_date
        for i in range(self.sim_days+1): 
            # Get data
            data = util.read_CD_data(date, self.CD_raw_data_direc, self.CD_processed_data_direc)
            if data is None:
                return  None

            if date == self.start_date and util.is_dst(date):
                # drop the first hour -> standard time
                data = data.iloc[60:]
                need_last_hour = True
            elif i == self.sim_days and need_last_hour:
                data = data.iloc[:60]

            cd = util.get_clean_CD_cycle_data(data)

            if use_avg_flow:
                targets.q_pc_target_on_in.extend(list(cd['Avg Q into cycle [MW]']))
            else:
                targets.q_pc_target_on_in.extend(list(cd['Q into cycle [MW]']))

            ## Cycle start up binary
            if ctrl_adj:
                pc_su = np.array([1 if x > 25. else 0 for x in cd['Gross Power [MW]']])  # Is cycle running?
            else:
                pc_su = np.array([1 if x > 5. else 0 for x in cd['Gross Power [MW]']])  # Is cycle running?
                
            pc_on.extend(pc_su.tolist())

            for ind in np.where(pc_su[:-1] != pc_su[1:])[0]:        # cycle changes condition
                if pc_su[ind + 1] == 1:
                    buffer = 20 #10
                    pc_su[int(ind - self.properties.startup_time*60 + buffer): ind+1] = 1   # push start-up forward

            targets.is_pc_su_allowed_in.extend(pc_su.tolist())   # Is cycle running?

            if set_rec_sb:
                # Receiver stand-by operation - attempt to control standby by tank temperature
                Nfbs = [1,2,3,4,5]
                tstep = 1/60
                dev = {}
                for Nfb in Nfbs:
                    dev[str(Nfb)] = [0]*Nfb
                    for j in range(Nfb, len(cd['Cold Tank Temp [C]']) - Nfb):
                        dev[str(Nfb)].append((cd['Cold Tank Temp [C]'].iloc[j+Nfb] - cd['Cold Tank Temp [C]'].iloc[j-Nfb])/2*Nfb*tstep)
                    dev[str(Nfb)].extend([0.]*Nfb)
                '''
                import matplotlib.pyplot as plt
                plt.figure()
                for Nfb in Nfbs:
                    plt.plot(dev[str(Nfb)], label = 'Nfb = ' + str(Nfb))
                plt.legend(loc = 'lower left')
                ax = plt.gca()
                ax2 = ax.twinx()
                ax2.plot(list(cd['Cold Tank Temp [C]']), label = 'Cold tank temp.')
                plt.legend(loc = 'lower right')
                plt.show()
                '''
                rec_sb = [1 if dTdt > 0.004 else 0 for dTdt in dev[str(1)]]
                targets.is_rec_sb_allowed_in.extend(rec_sb)

            for key in CD_plot_data:
                self.CD_data_for_plotting[key].extend(list(cd[key]))

            date += datetime.timedelta(days=1)  # advance a day
        
        max_pc_qin = (self.design.P_ref/self.design.design_eff)*self.properties.cycle_max_frac
        n = len(targets.is_pc_su_allowed_in)
        targets.q_pc_target_on_in = [targets.q_pc_target_on_in[j] if pc_on[j] == 1 else 0.0 for j in range(n)]
        targets.q_pc_target_su_in = [max_pc_qin if (targets.is_pc_su_allowed_in[j] == 1 and pc_on[j] == 0) else 0.0 for j in range(n)]
        targets.q_pc_max_in = [max_pc_qin for j in range(n)]     
        targets.is_pc_sb_allowed_in = [0 for j in range(n)]  # Cycle can not go into standby (using TES to keep cycle warm)
        targets.is_rec_su_allowed_in = [1 for j in range(n)]  # Receiver can always start up if available energy
        
        if not set_rec_sb:
            targets.is_rec_sb_allowed_in = [0 for j in range(n)]  # For now only allowing standby based on ssc determination from temperature threshold.  Specifying a value of 1 in any time period will force the receiver into standby regardless of temperature

        return targets
        
    def get_CD_NVE_day_ahead_schedule(self, date):
        targets = util.read_NVE_schedule(date, self.CD_raw_data_direc)
        if util.is_dst(date) and targets is not None: # First target in file is cumulative generation between 12am-1am PDT (11pm - 12am PST).  Ignore first point and read next-day file to define last point
            targets2 = util.read_NVE_schedule(date+datetime.timedelta(days=1), self.CD_raw_data_direc)
            targets = np.append(targets[1:], targets2[0])
        return targets

    
    def get_field_availability_adjustment(self, steps_per_hour, year):
        if self.control_field == 'ssc':
            if self.use_CD_measured_reflectivity:
                adjust = util.get_field_adjustment_from_CD_data(year, self.design.N_hel, self.properties.helio_reflectance*100, True, None, False)            
            else:
                adjust = (self.fixed_soiling_loss * 100 * np.ones(steps_per_hour*24*365))  

        elif self.control_field == 'CD_data':
            if self.use_CD_measured_reflectivity:
                adjust = util.get_field_adjustment_from_CD_data(year, self.design.N_hel, self.properties.helio_reflectance*100, True, None, True)
            else:
                refl = (1-self.fixed_soiling_loss) * self.properties.helio_reflectance * 100  # Simulated heliostat reflectivity
                adjust = util.get_field_adjustment_from_CD_data(year, self.design.N_hel, self.properties.helio_reflectance*100, False, refl, True)
 
        adjust = adjust.tolist()
        data_steps_per_hour = len(adjust)/8760  
        if data_steps_per_hour != steps_per_hour:
            adjust = util.translate_to_new_timestep(adjust, 1./data_steps_per_hour, 1./steps_per_hour)
        return adjust
    


    #--- Simulate flux maps
    def simulate_flux_maps(self):
        print ('Simulating flux maps')
        start = timeit.default_timer()
        D0 = vars(self.design)  
        D = D0.copy()
        D.update(vars(self.properties))
        D['time_steps_per_hour'] = self.ssc_time_steps_per_hour
        #D['solar_resource_file'] = self.ground_truth_weather_file
        D['solar_resource_data'] = self.ground_truth_weather_data
        D['time_start'] = 0.0
        D['time_stop'] = 1.0*3600  
        D['field_model_type'] = 2
        if self.is_debug:
            D['delta_flux_hrs'] = 4
            D['n_flux_days'] = 2
        R, state = ssc_wrapper.call_ssc(D, ['eta_map_out', 'flux_maps_for_import', 'A_sf'])  
        self.flux_maps.set_from_ssc(R)
        print('Time to simulate flux maps = %.2fs'%(timeit.default_timer() - start))
        return 
    

    
    #-------------------------------------------------------------------------
    #--- Run simulation
    def run(self, rerun_flux_maps = False):

        self.initialize()
            
        if self.flux_maps.A_sf_in == 0.0 or rerun_flux_maps:
            self.simulate_flux_maps()
            
        self.current_time = datetime.datetime(self.start_date.year, self.start_date.month, self.start_date.day)  # Start simulation at midnight (standard time) on the specifed day.  Note that annual arrays derived from CD data will not have "real" data after 11pm standard time during DST (unless data also exists for the following day)
        start_time = util.get_time_of_year(self.current_time)  # Time (sec) elapsed since beginning of year
        
        ntot = self.sim_days*24*self.ssc_time_steps_per_hour  # Total number of time points in solution
        retvars = self.default_ssc_return_vars()

        D0 = vars(self.design)  
        D = D0.copy()
        D.update(vars(self.properties))
        D.update(vars(self.flux_maps))
        D['ppa_multiplier_model'] = 1
        D['dispatch_factors_ts'] = self.price_data
        D['time_steps_per_hour'] = self.ssc_time_steps_per_hour
        D['is_rec_model_trans'] = self.use_transient_model
        D['is_rec_startup_trans'] = self.use_transient_startup
        D['rec_control_per_path'] = True
        D['solar_resource_data'] = self.ground_truth_weather_data
        D['field_model_type'] = 3
        D['eta_map_aod_format'] = False
        D['is_rec_to_coldtank_allowed'] = True
        D['rec_control_per_path'] = True

        #--- Initialize plant state at the start of the simulation period
        D.update(vars(self.plant_state))
        
        #--- Set field control parameters
        D['sf_adjust:hourly'] = self.get_field_availability_adjustment(self.ssc_time_steps_per_hour, self.current_time.year)
        if self.control_field == 'CD_data':
            D['is_rec_startup_trans'] = False
            D['rec_su_delay'] = 0.01   # Set receiver start time and energy to near zero to enforce CD receiver startup timing
            D['rec_qf_delay'] = 0.01
            

        #--- Set receiver control parameters
        if self.control_receiver == 'CD_data':
            D['is_rec_user_mflow'] = True

            mult = np.ones_like(self.CD_mflow_path2_data)
            if not self.use_CD_measured_reflectivity:  # Scale mass flow based on simulated reflectivity vs. CD actual reflectivity
                rho = (1-self.fixed_soiling_loss) * self.properties.helio_reflectance   # Simulated heliostat reflectivity
                CDavail = util.get_CD_soiling_availability(self.current_time.year, self.properties.helio_reflectance * 100)  # CD soiled / clean reflectivity (daily array)
                mult = rho*np.ones(365) / (CDavail*self.properties.helio_reflectance)  # Ratio of simulated / CD reflectivity
                mult = np.repeat(mult, 24*self.ssc_time_steps_per_hour)  # Annual array at ssc resolution

            D['rec_user_mflow_path_1'] = (self.CD_mflow_path2_data * mult).tolist()  # Note ssc path numbers are reversed relative to CD path numbers
            D['rec_user_mflow_path_2'] = (self.CD_mflow_path1_data * mult).tolist()
            D['is_rec_startup_trans'] = False
            D['rec_su_delay'] = 0.01   # Set receiver start time and energy to near zero to enforce CD receiver startup timing
            D['rec_qf_delay'] = 0.01            
        elif self.control_receiver == 'ssc_clearsky':
            D['rec_clearsky_fraction'] = 1.0
            D['rec_clearsky_model'] = 0
            D['rec_clearsky_dni'] = self.clearsky_data.tolist()
        elif self.control_receiver == 'ssc_actual_dni':
            D['rec_clearsky_fraction'] = 0.0


        #--- Set cycle control parameters and run simulation
        D['is_dispatch'] = 0    # Always disable dispatch optimization in ssc
        D['is_dispatch_targets'] = True if (self.is_optimize or self.control_cycle == 'CD_data') else False
        D['time_start'] = start_time
        D['time_stop'] = start_time + self.sim_days*24*3600   

        if self.is_optimize:
            R = self.run_rolling_horizon(D, self.sim_days*24)
        elif self.control_cycle == 'ssc_heuristic':
            if self.force_rolling_horizon:  # Run with rolling horizon (not necessary, but useful for debugging)
                R = self.run_rolling_horizon(D, self.sim_days*24)
            else:  # Run full time horizon in a single simulation
                R, state = ssc_wrapper.call_ssc(D, retvars, npts = ntot)  
        elif self.control_cycle == 'CD_data':
            retvars += vars(self.ssc_dispatch_targets).keys()
            self.ssc_dispatch_targets  = self.get_dispatch_targets_from_CD_actuals(use_avg_flow=True)
            D.update(vars(self.ssc_dispatch_targets))
            R, state = ssc_wrapper.call_ssc(D, retvars, npts = ntot)  
            
            
        self.results = R
        
        # Read NVE schedules (if not already read during rolling horizon calculations)
        if self.is_optimize == False and self.use_day_ahead_schedule and self.day_ahead_schedule_from == 'NVE':
            date = datetime.datetime(self.start_date.year, self.start_date.month, self.start_date.day) 
            for j in range(self.sim_days):
                newdate = date + datetime.timedelta(days = j)
                self.schedules.append(self.get_CD_NVE_day_ahead_schedule(newdate))

        self.calculate_revenue()
        self.calculate_day_ahead_penalty()
        self.calculate_startup_ramping_penalty()
        
        self.total_receiver_thermal = self.results['Q_thermal'].sum() * 1.e-3 * (1./self.ssc_time_steps_per_hour)
        self.total_cycle_gross = self.results['P_cycle'].sum() * 1.e-3 * (1./self.ssc_time_steps_per_hour)
        self.total_cycle_net = self.results['P_out_net'].sum() * 1.e-3 * (1./self.ssc_time_steps_per_hour)
        
        if self.save_results_to_file:
            self.save_results()            # Filename to save results: 
        

        return
            
    
    
    #-------------------------------------------------------------------------
    # Run simulation in a rolling horizon   
    #      The ssc simulation time resolution is assumed to be <= the shortest dispatch time step
    #      All dispatch time steps and time horizons are assumed to be an integer multiple of the ssc time step
    #      Time at which the weather forecast is updated coincides with the start of an optimization interval
    #      Time at which the day-ahead generation schedule is due coincides with the start of an optimization interval
    def run_rolling_horizon(self, D, total_horizon):
        start_time = util.get_time_of_year(self.current_time)  # Time (sec) elapsed since beginning of year
        start_hour = int(start_time / 3600)   # Time (hours) elapsed since beginning of year
        end_hour = start_hour + self.sim_days*24
        
        nph = int(self.ssc_time_steps_per_hour)         # Number of time steps per hour
        ntot = int(nph*total_horizon)                   # Total number of time points in full horizon
        napply = int(nph*self.dispatch_frequency)       # Number of ssc time points accepted after each solution 
        nupdate = int(total_horizon / self.dispatch_frequency)    # Number of update intervals
        startpt = int(start_hour*nph)                             # Start point in annual arrays
        
        
        sscstep = 3600/nph                              # ssc time step (s)
        nominal_horizon = int(self.dispatch_horizon*3600)  
        horizon_update = int(self.dispatch_horizon_update*3600)
        freq = int(self.dispatch_frequency*3600)    # Frequency of rolling horizon update (s)

        # Initialize results
        retvars = self.default_ssc_return_vars()
        retvars_disp = self.default_disp_stored_vars()
        if self.is_optimize:
            retvars += vars(self.ssc_dispatch_targets).keys()
        R = {k:np.zeros(ntot) for k in retvars}    
        for k in retvars_disp:
            R['disp_'+k] =np.zeros(ntot)

        self.disp_params_tracking = []
        self.disp_soln_tracking = []
        for j in range(nupdate):

  
            #--- Update input dictionary from current plant state
            D.update(vars(self.plant_state))
            
            #--- Set time horizon
            time = self.current_time
            tod = int(util.get_time_of_day(time))  # Current time of day (s)
            toy = int(util.get_time_of_year(time))  # Current time of year (s)
            D['time_start'] = toy
            startpt = int(toy/3600)*nph  # First point in annual arrays corresponding to this time
            horizon = nominal_horizon 
            if self.dispatch_horizon_update != self.dispatch_frequency:
                horizon = nominal_horizon  - int((j*freq) % horizon_update)   # Reduce dispatch horizon if re-optimization interval and horizon update interval are different
            
            # Restrict horizon to the end of the final simulation day (otherwise the dispatch often defers generation to the next day outside of the simulation window)
            if (toy+horizon)/3600 > end_hour:
                horizon = end_hour*3600 - toy  
                
            npts_horizon = int(horizon/3600 * nph)
            
            print (j, time, horizon/3600, npts_horizon, napply)
            

            #--- Update "forecasted" weather data (if relevant)
            if self.is_optimize and (tod == self.forecast_issue_time*3600):
                self.update_forecast_weather_data(time)

            #--- Update stored day-ahead generation schedule for current day (if relevant)
            if tod == 0 and self.use_day_ahead_schedule:  
                
                if self.day_ahead_schedule_from == 'calculated':
                    if j == 0:
                        self.schedules.append(None)  
                    else:
                        self.current_day_schedule = [s for s in self.next_day_schedule]
                        self.schedules.append(self.current_day_schedule)
                        
                elif self.day_ahead_schedule_from == 'NVE':
                    self.current_day_schedule = self.get_CD_NVE_day_ahead_schedule(time)
                    self.schedules.append(self.current_day_schedule)

                self.next_day_schedule = [0 for s in self.next_day_schedule]
                
            # Don't include day-ahead schedule if one hasn't been calculated yet, or if there is no NVE schedule available on this day
            include_day_ahead_in_dispatch = self.use_day_ahead_schedule
            if ((toy - start_time)/3600 < 24 and self.day_ahead_schedule_from == 'calculated') or (self.day_ahead_schedule_from == 'NVE' and self.current_day_schedule == None):  
                include_day_ahead_in_dispatch = False 


            #--- Run dispatch optimization (if relevant)
            if self.is_optimize:

                #--- Update prices for this horizon
                price = self.price_data[startpt:startpt+npts_horizon]  # $/MWh
                
                #--- Update weather to use in dispatch optimization for this optimization horizon
                self.weather_data_for_dispatch = self.update_dispatch_weather_data(
                    weather_data = self.weather_data_for_dispatch,
                    replacement_real_weather_data = self.ground_truth_weather_data,
                    replacement_forecast_weather_data = self.current_forecast_weather_data,
                    datetime = time,
                    total_horizon = horizon/3600.,
                    dispatch_horizon = self.dispatch_weather_horizon
                    )

                def estimates_for_dispatch_model(plant_design, toy, horizon, weather_data, N_pts_horizon, clearsky_data, start_pt):
                    D_est = plant_design.copy()
                    D_est['time_stop'] = toy + horizon
                    D_est['is_dispatch_targets'] = False
                    D_est['tshours'] = 100                      # Inflate TES size so that there is always "somewhere" to put receiver output
                    D_est['solar_resource_data'] = weather_data
                    D_est['is_rec_startup_trans'] = False
                    D_est['rec_su_delay'] = 0.001               # Simulate with no start-up time to get total available solar energy
                    D_est['rec_qf_delay'] = 0.001
                    retvars = ['Q_thermal', 'm_dot_rec', 'beam', 'clearsky', 'tdry', 'P_tower_pump', 'pparasi']
                    ssc_outputs, new_state = ssc_wrapper.call_ssc(D_est, retvars, npts = N_pts_horizon)
                    if ssc_outputs['clearsky'].max() < 1.e-3:         # Clear-sky data wasn't passed through ssc (ssc controlled from actual DNI, or user-defined flow inputs)
                        ssc_outputs['clearsky'] = clearsky_data[start_pt : start_pt + N_pts_horizon]
                    
                    return ssc_outputs

                #--- Run ssc for dispatch estimates: (using weather forecast time resolution for weather data and specified ssc time step)
                R_est = estimates_for_dispatch_model(
                    plant_design = D,
                    toy = toy,
                    horizon = horizon,
                    weather_data = self.weather_data_for_dispatch,
                    N_pts_horizon = npts_horizon,
                    clearsky_data = self.clearsky_data,
                    start_pt = startpt
                )

                # Some intermediate regression tests for refactoring purposes
                if j == 0 and toy + horizon == 24796800:
                    assert math.isclose(sum(list(R_est["Q_thermal"])), 230346, rel_tol=1e-4)
                    assert math.isclose(sum(list(R_est["m_dot_rec"])), 599416, rel_tol=1e-4)
                    assert math.isclose(sum(list(R_est["clearsky"])), 543582, rel_tol=1e-4)
                    assert math.isclose(sum(list(R_est["P_tower_pump"])), 2460.8, rel_tol=1e-4)

                def setup_dispatch_model(R_est,
                                         dispatch_params, plant_design, plant_state, nonlinear_model_time, use_linear_dispatch_at_night,
                                         clearsky_data, night_clearky_cutoff, dispatch_steplength_array, dispatch_steplength_end_time,
                                         disp_time_weighting, price, sscstep, avg_price, avg_price_disp_storage_incentive,
                                         avg_purchase_price, day_ahead_tol_plus, day_ahead_tol_minus,
                                         tod, current_day_schedule, day_ahead_pen_plus, day_ahead_pen_minus,
                                         dispatch_horizon, night_clearsky_cutoff, properties, dispatch_soln):

                    #--- Set dispatch optimization properties for this time horizon using ssc estimates
                    ##########
                    ##  There's already a lot of the dispatch_params member variables set here, which set_initial_state draws from
                    ##########
                    # Initialize dispatch model inputs
                    dispatch_params.set_dispatch_time_arrays(dispatch_steplength_array, dispatch_steplength_end_time,
                        dispatch_horizon, nonlinear_model_time, disp_time_weighting)
                    dispatch_params.set_fixed_parameters_from_plant_design(plant_design, properties)
                    dispatch_params.set_default_grid_limits()
                    dispatch_params.disp_time_weighting = disp_time_weighting
                    dispatch_params.set_initial_state(plant_design, plant_state)  # Set initial plant state for dispatch model
                    
                    # Update approximate receiver shutdown state from previous dispatch solution (not returned from ssc)
                    ursd = dispatch_soln.get_value_at_time(dispatch_params, freq/3600, 'ursd') 
                    yrsd = dispatch_soln.get_value_at_time(dispatch_params, freq/3600, 'yrsd') 
                    dispatch_params.set_approximate_shutdown_state_parameters(plant_state, ursd = ursd, yrsd = yrsd)  # Set initial state parameters related to shutdown from dispatch model (because this cannot be derived from ssc)

                    nonlinear_time = nonlinear_model_time # Time horizon for nonlinear model (hr)
                    if use_linear_dispatch_at_night:
                        endpt = int((toy + nonlinear_time*3600) / sscstep)  # Last point in annual arrays at ssc time step resolution corresponding to nonlinear portion of dispatch model
                        if clearsky_data[startpt:endpt].max() <= night_clearsky_cutoff:
                            nonlinear_time = 0.0

                    dispatch_params.set_dispatch_time_arrays(dispatch_steplength_array, dispatch_steplength_end_time, horizon/3600., nonlinear_time, disp_time_weighting)
                    dispatch_params.set_default_grid_limits()
                    dispatch_params.P = util.translate_to_variable_timestep([p/1000. for p in price], sscstep/3600., dispatch_params.Delta)  # $/kWh
                    dispatch_params.avg_price = avg_price/1000.
                    dispatch_params.avg_price_disp_storage_incentive = avg_price_disp_storage_incentive / 1000.  # $/kWh  # Only used in storage inventory incentive -> high values cause the solutions to max out storage rather than generate electricity
                    dispatch_params.avg_purchase_price = avg_purchase_price/1000    # $/kWh 
                    dispatch_params.day_ahead_tol_plus = day_ahead_tol_plus*1000    # kWhe
                    dispatch_params.day_ahead_tol_minus = day_ahead_tol_minus*1000  # kWhe

                    dispatch_params.set_estimates_from_ssc_data(plant_design, R_est, sscstep/3600.) 
                    
                    
                    #--- Set day-ahead schedule in dispatch parameters
                    if include_day_ahead_in_dispatch:  
                        day_ahead_horizon = 24 - int(tod/3600)   # Number of hours of day-ahead schedule to use.  This probably only works in the dispatch model if time horizons are updated at integer multiples of an hour
                        use_schedule = [current_day_schedule[s]*1000 for s in range(24-day_ahead_horizon, 24)]   # kWhe
                        dispatch_params.set_day_ahead_schedule(use_schedule, day_ahead_pen_plus/1000, day_ahead_pen_minus/1000)
                        

                    #--- Run dispatch optimization and set ssc dispatch targets
                    disp_in = dispatch_params.copy_and_format_indexed_inputs()     # dispatch.DispatchParams object

                    return disp_in


                #--- Set dispatch optimization properties for this time horizon using ssc estimates
                disp_in = setup_dispatch_model(
                    R_est = R_est,
                    dispatch_params = self.dispatch_params,
                    dispatch_steplength_array = self.dispatch_steplength_array,
                    dispatch_steplength_end_time = self.dispatch_steplength_end_time,
                    dispatch_horizon = self.dispatch_horizon,
                    plant_design = self.design,
                    plant_state = self.plant_state,
                    nonlinear_model_time = self.nonlinear_model_time,
                    use_linear_dispatch_at_night = self.use_linear_dispatch_at_night,
                    clearsky_data = self.clearsky_data,
                    night_clearky_cutoff = self.night_clearsky_cutoff,
                    disp_time_weighting = self.disp_time_weighting,
                    price = price,
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
                    properties = self.properties,
                    dispatch_soln = self.dispatch_soln
                )
                
                def run_dispatch_model(disp_in, include, dispatch_soln, transition=0):
                    disp_out = run_phase_one.run_dispatch(disp_in, include, disp_in.start, disp_in.stop, transition=0)
                    if disp_out is not False:  
                        dispatch_soln.set_from_dispatch_outputs(disp_out)
                        # D['is_dispatch_targets'] = True
                    else:  # Infeasible solution was returned, revert back to running ssc without dispatch targets
                        print ('Infeasible dispatch solution')
                        return None
                        # D['is_dispatch_targets'] = False

                    return dispatch_soln

                def get_day_ahead_schedule(day_ahead_schedule_steps_per_hour, Delta, Delta_e, net_electrical_output, day_ahead_schedule_time):
                    print ('Storing day-ahead schedule')
                    day_ahead_step = 1./day_ahead_schedule_steps_per_hour  # Time step for day-ahead schedule (hr)
                    disp_steps = np.array(Delta)
                    wnet = np.array(net_electrical_output) / 1000.   # Net electricity to the grid (MWe) at dispatch time steps
                    inds = np.where(np.array(Delta_e) > (24 - day_ahead_schedule_time))[0]    # Find time points corresponding to the 24-hour period starting at midnight on the next day
                    if len(inds) >0:  # Last-day simulation won't have any points in day-ahead period
                        diff = np.abs(disp_steps[inds] - day_ahead_step)
                        next_day_schedule = wnet[inds]
                        if diff.max() > 1.e-3:
                            next_day_schedule = util.translate_to_fixed_timestep(wnet[inds], disp_steps[inds], day_ahead_step) 
                        return next_day_schedule
                    else:
                        return None

                def get_weather_at_day_ahead_schedule(weather_data_for_dispatch, startpt, npts_horizon):
                    return {k:weather_data_for_dispatch[k][startpt:startpt+npts_horizon] for k in ['dn', 'tdry', 'wspd']}



                include = {"pv": False, "battery": False, "persistence": False, "force_cycle": False, "op_assumptions": False,
                           "signal":include_day_ahead_in_dispatch, "simple_receiver": False}
                    
                self.dispatch_soln = run_dispatch_model(disp_in, include, self.dispatch_soln)

                if self.dispatch_soln is not None:
                    if self.use_day_ahead_schedule and self.day_ahead_schedule_from == 'calculated' and tod/3600 == self.day_ahead_schedule_time:
                        self.next_day_schedule = get_day_ahead_schedule(
                            day_ahead_schedule_steps_per_hour = self.day_ahead_schedule_steps_per_hour,
                            Delta = self.dispatch_params.Delta,
                            Delta_e = self.dispatch_params.Delta_e,
                            net_electrical_output = self.dispatch_soln.net_electrical_output,
                            day_ahead_schedule_time = self.day_ahead_schedule_time
                            )

                        weather_at_day_ahead_schedule = get_weather_at_day_ahead_schedule(self.weather_data_for_dispatch, startpt, npts_horizon)
                        self.weather_at_schedule.append(weather_at_day_ahead_schedule)  # Store weather used at the point in time the day ahead schedule was generated

                    #--- Set ssc dispatch targets
                    ssc_dispatch_targets = ssc_wrapper.DispatchTargets()
                    ssc_dispatch_targets.set_from_dispatch_solution(self.design, self.properties, self.dispatch_params, self.dispatch_soln, self.plant_state, sscstep/3600., freq/3600.)
                    D.update(vars(ssc_dispatch_targets))

                    # Regression tests for dispatch model outputs (ssc_dispatch_targets)
                    if j == 0 and toy + horizon == 24796800:
                        assert hash(tuple(ssc_dispatch_targets.is_pc_sb_allowed_in)) == -4965923453060612375
                        assert hash(tuple(ssc_dispatch_targets.is_pc_su_allowed_in)) == -4965923453060612375
                        assert hash(tuple(ssc_dispatch_targets.is_rec_sb_allowed_in)) == -4965923453060612375
                        assert hash(tuple(ssc_dispatch_targets.is_rec_su_allowed_in)) == -4965923453060612375
                        assert hash(tuple(ssc_dispatch_targets.q_pc_max_in)) == -709626543671595165
                        assert hash(tuple(ssc_dispatch_targets.q_pc_target_on_in)) == -4965923453060612375
                        assert hash(tuple(ssc_dispatch_targets.q_pc_target_su_in)) == -4965923453060612375


                else:  # Infeasible solution was returned, revert back to running ssc without dispatch targets
                    pass
                    # print ('Infeasible dispatch solution')
                    # D['is_dispatch_targets'] = False
                    # if self.store_full_dispatch_solns:  # Store complete dispatch solutions for debugging
                    #     self.disp_params_tracking.append(deepcopy(self.dispatch_params))
                    #     self.disp_soln_tracking.append(None)
                    #     self.plant_state_tracking.append(deepcopy(self.plant_state))
                    #     self.infeasible_count+=1






            #--- Run ssc and collect results
            D['time_stop'] = toy+freq
            Rsub, new_plant_state = ssc_wrapper.call_ssc(D, retvars, plant_state_pt = napply-1, npts = napply) 
            new_plant_state.update_persistence(self.plant_state, Rsub, sscstep/3600.)
            self.plant_state = new_plant_state
            for k in Rsub.keys():
                R[k][j*napply:(j+1)*napply] = Rsub[k][0:napply]
                

            #--- Add dispatch model solution to results
            if self.is_optimize and self.dispatch_soln is not None:
                Rdisp = self.dispatch_soln.get_solution_at_ssc_steps(self.dispatch_params, sscstep/3600., freq/3600.)
                for k in retvars_disp:
                    if k in Rdisp.keys():
                        R['disp_'+k][j*napply:(j+1)*napply] = Rdisp[k]
                    

            #--- Update current time
            self.current_time = self.current_time + datetime.timedelta(seconds = freq)
            
        return R
            

    # Calculate revenue
    def calculate_revenue(self):
        nph = int(self.ssc_time_steps_per_hour)
        ndays = self.sim_days
        start = datetime.datetime(self.start_date.year, self.start_date.month, self.start_date.day) 
        startpt = int(util.get_time_of_year(start)/3600) * nph   # First point in annual arrays         
        price = np.array(self.price_data[startpt:startpt+ndays*24*nph])
        mult = price / price.mean()   # Pricing multipliers
        
        net_gen = self.results['P_out_net']
        inds_sell = np.where(net_gen > 0.0)[0]
        inds_buy = np.where(net_gen < 0.0)[0]
        rev = (net_gen[inds_sell] * mult[inds_sell] * self.avg_price).sum() * (1./self.ssc_time_steps_per_hour)   # Revenue from sales ($)
        rev += (net_gen[inds_buy] * mult[inds_buy] * self.avg_purchase_price).sum() * (1./self.ssc_time_steps_per_hour) # Electricity purchases ($)
        self.revenue = rev        
        return 

    
    # Calculate penalty for missing day-ahead schedule (assuming day-ahead schedule step is 1-hour for now)
    def calculate_day_ahead_penalty(self):
        ndays = self.sim_days
        nph = int(self.ssc_time_steps_per_hour)

        self.day_ahead_diff = {k:np.zeros((ndays, 24)) for k in ['ssc', 'disp', 'disp_raw']}
        self.day_ahead_penalty = {k:np.zeros((ndays, 24)) for k in ['ssc', 'disp', 'disp_raw']}
        self.day_ahead_diff_over_tol_plus = {k:0.0 for k in ['ssc', 'disp', 'disp_raw']}  
        self.day_ahead_diff_over_tol_minus = {k:0.0 for k in ['ssc', 'disp', 'disp_raw']}
        
        self.day_ahead_diff_ssc_disp_gross = np.zeros((ndays, 24))
                
        # Calculate penalty from ssc or dispatch results (translated to ssc time steps)
        for d in range(ndays):
            if len(self.schedules) > d and self.schedules[d] is not None:  # Schedule exists
                for j in range(24):  # Hours per day
                    target = self.schedules[d][j]       # Target generation during the schedule step
                    p = d*24*nph + j*nph                # First point in result arrays from ssc solutions
                    
                    wnet = {k:0.0 for k in ['ssc', 'disp', 'disp_raw']}
                    wnet['ssc'] = self.results['P_out_net'][p:p+nph].sum() * 1./nph                             # Total generation from ssc during the schedule step (MWhe)
                    if 'disp_net_electrical_output' in self.results.keys():
                        wnet['disp'] = self.results['disp_net_electrical_output'][p:p+nph].sum() * 1./nph * 1.e-3   # Total generation from dispatch solution during the schedule step (MWhe)
                    
                    self.day_ahead_diff_ssc_disp_gross[d,j] = wnet['ssc'] - wnet['disp']
                    
                    # Calculate generation directly from dispatch schedule before interpolation
                    if len(self.disp_soln_tracking)>0:  # Dispatch solutions were saved
                        i = d*24+j
                        delta_e = self.disp_params_tracking[i].Delta_e
                        delta = self.disp_params_tracking[i].Delta 
                        wdisp = self.disp_soln_tracking[i].net_electrical_output/1000.  # Net energy sold to grid (MWe)
                        inds = np.where(np.array(delta_e) <= 1.0)[0]
                        wnet['disp_raw'] = sum([wdisp[i]*delta[i] for i in inds])  # MWhe cumulative generation 
                        
                    for k in self.day_ahead_diff.keys():
                        self.day_ahead_diff[k][d,j] = wnet[k] - target
                        
                        if not self.day_ahead_ignore_off or target>0.0 or wnet[k]>0:  # Enforce penalties for missing schedule
                            self.day_ahead_diff[k][d,j] = wnet[k] - target
                            if self.day_ahead_diff[k][d,j] > self.day_ahead_tol_plus:
                                self.day_ahead_penalty[k][d,j] = self.day_ahead_diff[k][d,j] * self.day_ahead_pen_plus
                                self.day_ahead_diff_over_tol_plus[k] += self.day_ahead_diff[k][d,j]
                            elif self.day_ahead_diff[k][d,j] < self.day_ahead_tol_minus:
                                self.day_ahead_penalty[k][d,j] = (-self.day_ahead_diff[k][d,j]) * self.day_ahead_pen_minus
                                self.day_ahead_diff_over_tol_minus[k] += self.day_ahead_diff[k][d,j]
                                
                                

        self.day_ahead_penalty_tot = {k:self.day_ahead_penalty[k].sum() for k in self.day_ahead_diff.keys()}  # Total penalty ($)
        self.day_ahead_diff_tot = {k:self.day_ahead_diff[k].sum() for k in self.day_ahead_diff.keys()}
        return



    def find_starts(self, q_start, q_on):
        n = len(q_start)
        n_starts, n_starts_attempted = [0, 0]
        is_starting = False
        for j in range(1,n):
            
            if q_start[j] > 1.e-3 and q_start[j-1]<1.e-3:
                is_starting = True
                n_starts_attempted +=1
            
            if is_starting:
                if q_on[j] > 1.e-3:  # Startup completed
                    n_starts += 1
                    is_starting = False
                elif q_start[j] < 1.e-3: # Startup abandoned
                    is_starting = False    
                    
        return n_starts, n_starts_attempted
    
    
    def calculate_startup_ramping_penalty(self):

        #self.n_starts_rec = np.logical_and(self.results['q_startup'][1:]>1.e-3, self.results['q_startup'][0:-1] < 1.e-3).sum()  # Nonzero startup energy in step t and zero startup energy in t-1
        #self.n_starts_cycle =  np.logical_and(self.results['q_dot_pc_startup'][1:]>1.e-3, self.results['q_dot_pc_startup'][0:-1] < 1.e-3).sum()
        
        self.n_starts_rec, self.n_starts_rec_attempted =  self.find_starts(self.results['q_startup'], self.results['Q_thermal'])

        qpb_on = self.results['q_pb']   # Cycle thermal power includes startup
        inds_off = np.where(self.results['P_cycle']<1.e-3)[0]
        qpb_on[inds_off] = 0.0
        self.n_starts_cycle, self.n_starts_cycle_attempted = self.find_starts(self.results['q_dot_pc_startup'], qpb_on)

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
        
        self.startup_ramping_penalty = self.n_starts_rec*self.properties.Crsu + self.n_starts_cycle*self.properties.Ccsu 
        self.startup_ramping_penalty += self.cycle_ramp_up*self.properties.C_delta_w*1000 + self.cycle_ramp_down*self.properties.C_delta_w*1000
        return
        
      
    def default_ssc_return_vars(self):
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
    
    
    def save_results(self):
        # Save time series results
        filename = self.results_file
        array = np.transpose(np.array([v for v in self.results.values()]))
        n = array.shape[0]
        times = np.reshape(np.arange(n)* 1./self.ssc_time_steps_per_hour, (n,1))
        array = np.append(times, array, 1)
        header = ','.join(['Time']+list(self.results.keys()))
        np.savetxt(filename+'_results.csv', array, delimiter = ',', header = header, comments = '', fmt = '%.6f')
        
        # Save schedules
        if self.use_day_ahead_schedule:
            nday = len(self.schedules)
            nperday = int(24*self.day_ahead_schedule_steps_per_hour)
            array = np.zeros((nperday, nday))
            for d in range(nday):
                array[:,d] = self.schedules[d] if self.schedules[d] is not None else np.nan * np.ones(nperday)
            times = np.reshape(np.arange(nperday)* 24/nperday, (nperday,1))
            array = np.append(times, array, 1)
            header = ','.join(['Hour']+['Day %d'%d for d in range(nday)])
            np.savetxt(filename+'_schedules.csv', array, delimiter = ',', header = header, comments = '', fmt = '%.6f')
            
            
        # Save summary output   
        keys = ['revenue', 'startup_ramping_penalty', 'n_starts_rec', 'n_starts_rec_attempted', 'n_starts_cycle', 'n_starts_cycle_attempted',
                'cycle_ramp_up', 'cycle_ramp_down', 'total_receiver_thermal', 'total_cycle_gross', 'total_cycle_net']        
        data = [getattr(self,k) for k in keys] 
        data += [self.day_ahead_penalty_tot[k] for k in ['ssc', 'disp']]
        data += [self.day_ahead_diff_over_tol_plus[k] for k in ['ssc', 'disp']]
        data += [self.day_ahead_diff_over_tol_minus[k] for k in ['ssc', 'disp']]
        
        keys += ['day_ahead_penalty_tot (ssc)', 'day_ahead_penalty_tot (disp)']
        keys += ['day_ahead_diff_over_tol_plus (ssc)', 'day_ahead_diff_over_tol_plus (disp)']
        keys += ['day_ahead_diff_over_tol_minus (ssc)', 'day_ahead_diff_over_tol_minus (disp)']
        
        np.savetxt(filename+ '_summary.csv', np.reshape(data, (1,len(data))), header = ','.join(keys), delimiter = ',', comments = '', fmt = '%.6f')

        return



    def load_results_from_file(self, filename):
        self.results = {}
        self.schedules = []
        cols = np.genfromtxt(filename +'_results.csv', delimiter = ',', max_rows = 1, dtype = str)
        data = np.genfromtxt(filename+'_results.csv', delimiter = ',', skip_header = 1)
        for c in range(len(cols)):
            self.results[cols[c]] = data[:,c]
            
        # Read summary
        cols = np.genfromtxt(filename +'_summary.csv', delimiter = ',', max_rows = 1, dtype = str)
        data = np.genfromtxt(filename+'_summary.csv', delimiter = ',', skip_header = 1)        
        self.day_ahead_penalty_tot = {}
        self.day_ahead_diff_over_tol_plus = {}
        self.day_ahead_diff_over_tol_minus = {}
        
        for c in range(len(cols)):
            if cols[c] not in ['day_ahead_penalty_tot (ssc)', 'day_ahead_penalty_tot (disp)', 
                               'day_ahead_diff_over_tol_plus (ssc)', 'day_ahead_diff_over_tol_plus (disp)', 
                               'day_ahead_diff_over_tol_minus (ssc)', 'day_ahead_diff_over_tol_minus (disp)']:
                setattr(self, cols[c], data[c])
            else:
                name = cols[c].split(' ')[0]
                k = cols[c].split(' ')[1][1:-1]
                if name == 'day_ahead_penalty_tot':
                    self.day_ahead_penalty_tot[k] = data[c]
                elif name == 'day_ahead_diff_over_tol_plus':
                    self.day_ahead_diff_over_tol_plus[k] = data[c]
                elif name == 'day_ahead_diff_over_tol_minus':
                    self.day_ahead_diff_over_tol_minus[k] = data[c]                    

        # Read schedules
        try:
            data = np.genfromtxt(filename+'_schedules.csv', delimiter = ',', skip_header = 1)
            for d in range(1,data.shape[1]):
                self.schedules.append(None if np.isnan(data[0,d]) else data[:,d])
        except:
            print ('Unable to read day-ahead schedules')
        
        return



    
    
        
        

    
    
    

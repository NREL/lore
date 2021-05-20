import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from math import ceil, pi, log, isnan
import numpy as np
from copy import deepcopy
import pyomo.environ as pe
from pyomo.opt import TerminationCondition
import datetime

try:
    import librtdispatch.util as util
    import librtdispatch.dispatch_model as dispatch_model
except:     # if called from case_study
    from librtdispatch import util
    from librtdispatch import dispatch_model


dispatch_wrap_params = {
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
}


class DispatchParams:
    def __init__(self):
        
        #---------------------------------------------------------------------
        # Indexing
        self.T = 0           # Number of time steps in time-indexed parameters
        self.start = 0       # First index of the problem
        self.stop = 0        # Last index of the problem
        self.transition = 0  # Index of the transition between the models
        self.nc = 1          # Number of linear regions in HTF pumping power through cycle regression
        self.nfw = 1         # Number of linear regions in feed water pumping power through cycle regression
        
        #---------------------------------------------------------------------
        # Piecewise-linear indexed parameters
        self.Pc = []        # Slope of the HTF pumping power through the cycle regression region i [kWe / kg/s]
        self.Bc = []        # Intercept of the HTF pumping power through the cycle regression region i [kWe]
        self.Pfw = []       # Slope of the feed water pumping power through the cycle regression region i [kWe / kg/s]
        self.Bfw = []       # Intercept of the feed water pumping power through the cycle regression region i [kWe]
        
        #--------------------------------------------------------------------
        # Field and receiver parameters
        self.Drsu = 0.0        # Minimum time to start the receiver (hr)
        self.Drsd = 0.0        # Minimum time to shut down the receiver (hr)
        self.Er = 0.0          # Required energy expended to start receiver (kWht)
        self.Qrl = 0.0         # Minimum operational thermal power delivered by receiver (kWt)
        self.Qrsb = 0.0        # Required thermal power for receiver standby (kWt)
        self.Qrsd = 0.0        # Required thermal power for receiver shut down (kWht?)
        self.Qru = 0.0         # Allowable power per period for receiver start-up (kWt)
        self.mdot_r_min = 0.0  # Minimum mass flow rate of HTF to the receiver (kg/s)
        self.mdot_r_max = 0.0  # Maximum mass flow rate of HTF to the receiver (kg/s)
        self.T_rout_min = 0.0  # Minimum allowable receiver outlet T (C)
        self.T_rout_max = 0.0  # Maximum allowable receiver outlet T (C)

        #--------------------------------------------------------------------
        # TES parameters
        self.Eu = 0.0           # Thermal energy storage capacity (kWht)
        self.Cp = 0.0           # Specific heat of HTF (kJ/kg/K)
        self.mass_cs_min = 0.0  # Minimum mass of HTF in cold storage (kg)
        self.mass_cs_max = 0.0  # Maximum mass of HTF in cold storage (kg)
        self.mass_hs_min = 0.0  # Minimum mass of HTF in hot storage (kg)
        self.mass_hs_max = 0.0  # Maximum mass of HTF in hot storage (kg)
        self.T_cs_min = 0.0     # Minimum HTF T in cold storage (C)
        self.T_cs_max = 0.0     # Maximum HTF T in cold storage (C)
        self.T_hs_min = 0.0     # Minimum HTF T in cold storage (C)
        self.T_hs_max = 0.0     # Maximum HTF T in hot storage (C)
        self.T_cs_des = 0.0     # Design point cold storage temperature (C)
        self.T_hs_des = 0.0     # Design point hot storage temperature (C)
        
        #--------------------------------------------------------------------
        # Cycle parameters
        self.Ec = 0.0           # Required energy expended to start cycle (kWht)
        self.Ew = 0.0           # Required energy expended to warm-start the cycle (kWht)
        self.eta_des = 0.0      # Cycle design point efficiency 
        self.etap = 0.0         # Slope of linear approximation to power cycle performance curve (kWe/kWt)
        self.Qb = 0.0           # Cycle standby thermal power consumption (kWt)
        self.Qc = 0.0           # Allowable power per period for cycle startup (kWt)
        self.Ql = 0.0           # Minimum operational thermal power input to cycle (kWt)
        self.Qu = 0.0           # Cycle thermal power capacity (kWt)
        self.kl = 0.0           # Change in lower bound of cycle thermal load due to hot storage temperature (kWt/C)
        self.ku = 0.0           # Change in upper bound of cycle thermal load due to hot storage temperature (kWt/C)
        self.Wdot_design = 0.0  # Design point electrical output of the power cycle (kWe)
        self.Wdot_p_max = 25000. # Power purchase required to cover all parasitic loads [kWe]
        self.Wdotl = 0.0        # Minimum cycle electric power output (kWe)
        self.Wdotu = 0.0        # Cycle electric power rated capacity (kWe)
        self.delta_T_design = 0.0 # Design point temperature change of HTF across the SGS model (C)
        self.delta_T_max = 0.0   # Max temperature change of HTF across the SGS model (C)
        self.mdot_c_design = 0.0 # Design point mass flow rate of HTF to the power cycle (kg/s)
        self.mdot_c_min = 0.0    # Minmium mass flow rate of HTF to the power cycle (kg/s)
        self.mdot_c_max = 0.0    # Maximum mass flow rate of HTF to  the power cycle (kg/s)
        self.T_cin_design = 0.0  # HTF design point power cycle inlet temperature (C)
        self.T_cout_min = 0.0    # HTF design point power cycle inlet temperature (C)
        self.T_cout_max = 0.0    # HTF design point power cycle inlet temperature (C)
        self.W_delta_plus = 0.0  # Power cycle ramp-up designed limit (kWe/h)
        self.W_delta_minus = 0.0 # Power cycle ramp-down designed limit (kWe/h)
        self.W_v_plus = 0.0      # Power cycle ramp-up violation limit (kWe/h)
        self.W_v_minus = 0.0     # Power cycle ramp-down violation limit (kWe/h)
        self.Yu = 0.0            # Minimum required cycle uptime (h)
        self.Yd = 0.0            # Minimum required cycle downtime (h)

        #--------------------------------------------------------------------
        # Parastic loads
        self.Ehs = 0.0         # Heliostat field startup or shut down parasitic loss (kWhe)
        self.Wh_track = 0.0    # Heliostat tracking parasitic loss (kWe)
        self.Wh_comm = 0.0     # Heliostat field communication parasitic loss (kWe)
        self.Lr = 0.0          # Receiver pumping power per unit power produced (kWe/kWt)
        self.Pr = 0.0          # Receiver pumping power per unit mass flow (kW/kg/s)
        self.Wht_full = 0.0    # Tower piping heat trace full-load parasitic load (kWe)  
        self.Wht_part = 0.0    # Tower piping heat trace full-load parasitic load (kWe)  
        self.Lc = 0.0          # Cycle HTF pumping power per unit energy expended (kWe/kWt)
        self.Wb = 0.0          # Power cycle standby operation parasitic load (kWe)
        self.Wc = 0.0          # Power cycle operation parasitic load (kWe)

        #--------------------------------------------------------------------
        # Cost parameters
        self.alpha = 1.0        # Conversion factor between unitless and monetary values ($)
        self.Crec = 0.0         # Operating cost of heliostat field and receiver
        self.Crsu = 0.0         # Penalty for receiver cold start-up ($/start)
        self.Crhsp = 0.0        # Penalty for receiver hot start-up ($/start)
        self.Cpc = 0.0          # Operating cost of power cycle ($/kWh)
        self.Ccsu = 0.0         # Penalty for power cycle cold start-up ($/start)
        self.Cchsp = 0.0        # Penalty for power cycle hot start-up ($/start)
        self.C_delta_w = 0.0    # Penalty for change in power cycle production ($/kWe)
        self.C_v_w = 0.0        # Penalty for change in power cycle production (beyond designated limits) ($/kWe)
        self.Ccsb = 0.0         # Operating cost of power cycle standby operation ($/kWh)
        
        #--------------------------------------------------------------------
        # Other parameters
        self.alpha_b = 0.0
        self.alpha_T = 0.0
        self.alpha_m = 0.0
        self.beta_b = 0.0
        self.beta_T = 0.0
        self.beta_m = 0.0
        self.beta_mT = 0.0
        
        #---------------------------------------------------------------------
        # Miscellaneous Parameters 
        self.disp_time_weighting = 0.99
        self.avg_price = 1.0
        
        #---------------------------------------------------------------------
        # Initial conditions
        self.s0 = 0.0       # Initial TES charge state (kWht)
        self.ucsu0 = 0.0    # Initial cycle start-up energy inventory (kWht)
        self.ursu0 = 0.0    # Initial receiver start-up energy inventory (kWht)
        self.ursd0 = 0.0    # Initial receiver shut down energy inventory (kWht)
        self.wdot0 = 0.0    # Initial cycle electricity generation (kWe)
        self.mass_cs0 = 0.0 # Initial mass of HTF in cold storage (kg)
        self.mass_hs0 = 0.0 # Initial mass of HTF in hot storage (kg)
        self.T_cs0 = 0.0    # Initial temperature of HTF in cold storage (C)
        self.T_hs0 = 0.0    # Initial temperature of HTF in hot storage (C)
        
        self.yr0 = 0        # Is receiver "on"?
        self.yrsb0 = 0      # Is receiver in standby?
        self.yrsd0 = 0      # IS receiver in shutdown?
        self.yrsu0 = 0      # is receiver starting up?
        
        self.y0 = 0         # Is cycle generating electric power?
        self.ycsb0 = 0      # Is cycle in standby?
        self.ycsu0 = 0      # Is cycle starting up?
        
        self.drsd0 = 0.0   # Duration that receiver has been in shutdown mode before the problem horizon (h)
        self.drsu0 = 0.0   # Duration that receiver has been starting up before the problem horizon (h)
        self.Yu0 = 0.0     # Duration that cycle has been generating electric power (h)
        self.Yd0 = 0.0     # Duration that cycle has been shut down (h)
        
        #---------------------------------------------------------------------
        # Time-indexed parameters 
        self.Delta = []         # Time step length in period t (hr)
        self.Delta_e = []       # Cumulative time elapased at the end of period t (hr)
        self.D = []             # Time-weighted discount factor in period t
        self.P = []             # Electricity sales price ($/kWhe)
        self.Wdotnet = []       # Net electricity generation limit (kWe)
        self.W_u_plus = []      # Maximum power production in period t when starting generation (kW)
        self.W_u_minus = []     # Maximum power production in period t when stopping generation in period t+1 (kW)
        
        
        # Time-indexed parameters derived from ssc estimates
        self.Qin = []       # Expected thermal generation from solar field and receiver (kWt) 
        self.delta_rs = []  # Estimated fraction of period t required for receiver start-up
        self.delta_cs = []  # Estimated fraction of period t required for cycle start-up
        self.etaamb = []    # Expected power cycle efficiency ambient adjustment 
        self.etac = []      # Expected condenser parasitic loss coefficient 
        self.F = []         # Ratio of actual to clear-sky CSP heliostat field power
        self.Q_cls = []     # Expected thermal generation from solar field and receiver at clear-sky DNI (kWt) 
        
        # Time-indexed parameters derived from ssc estimates when running cases with fixed receiver operation
        self.P_field_rec = []  # Field/receiver parasitics (pumping power and heliostat drives) (kWe)
        
        #--------------------------------------------------------------------
        # Grid signal parameters
        self.num_signal_hours = 0   # Number of hours in the day-ahead signal
        self.G = []                 # Grid signal per hour (cumulative generation over the hour) (kWhe)
        self.Cg_plus = []           # Over-production penalty ($/kWhe)
        self.Cg_minus = []          # Under-production penalty ($/kWhe)
        
        #---- Net inputs
        self.avg_purchase_price = 0   # Average electricity purchase price ($/kWh) 
        self.day_ahead_tol_plus = 0   # Tolerance for over-production relative to day-ahead schedule before incurring penalty (kWhe)
        self.day_ahead_tol_minus= 0   # Tolerance for under-production relative to day-ahead schedule before incurring penalty (kWhe)

        return


    # Set dispatch time arrays  and time weighting factors
    def set_dispatch_time_arrays(self, steps, end_times, horizon, nonlinear_time = 0.0, timewt = 0.999):
        self.Delta = []
        self.Delta_e = []
        n = len(steps)
        if len(end_times) != n or end_times[-1] < horizon:
            print ('Error: Inconsistent specification of dispatch time step length, time horizon, and end points')
            return
        t = 0.0
        for j in range(n):
            dt = steps[j]   # min
            time_end = min(end_times[j], horizon) * 60  # min
            ns = int(ceil((time_end - t*60) / dt))
            for i in range(ns):
                self.Delta.append(dt/60.)
                self.Delta_e.append(t+dt/60.)
                t += dt/60.
            if (t > horizon):
                self.Delta[-1] -= (t - horizon)
                self.Delta_e[-1] = horizon
                break
        
        n = len(self.Delta)
        self.T = n
        self.start = 1  
        self.stop = n  
        
        if self.Delta_e[-1] < nonlinear_time:
            self.transition = n
        else:
            self.transition = np.where(np.array(self.Delta_e) >= nonlinear_time)[0][0]   # Last index of non-linear formulation
        self.set_time_weighting_factor(timewt)
        
                
        # Time-indexed parameters that are only dependent on fixed inputs and time step arrays
        self.W_u_plus = [(self.Wdotl + self.W_delta_plus*0.5*dt) for dt in self.Delta]
        self.W_u_minus = [(self.Wdotl + self.W_delta_minus*0.5*dt) for dt in self.Delta]

        return
    
    # Set fixed dispatch parameters from plant design and operating property specifications (call this after setting up time step arrays)
    def set_fixed_parameters_from_plant_design(self, plant):
        q_pb_design = plant.get_cycle_thermal_rating()  #MWt
        m_pb_design = plant.get_cycle_design_mass_flow()   # kg/s
        q_rec_design = plant.design['Qrec']  # MWt
        m_rec_design = plant.get_receiver_design_mass_flow() #kg/s
        nhel = plant.get_number_of_heliostats()
        
        m_active_hot_max, m_active_cold_max, m_inactive_hot, m_inactive_cold = plant.get_storage_mass()
        
        # Receiver parameters
        self.Drsu = plant.design['rec_su_delay']
        self.Drsd = plant.design['rec_sd_delay']
        self.Er = plant.design['rec_qf_delay'] * q_rec_design * 1000. 
        self.Qrl = plant.design['f_rec_min'] * q_rec_design * 1000.
        self.Qrsb = plant.design['q_rec_standby_fraction'] * q_rec_design * 1000.
        self.Qrsd = plant.design['q_rec_shutdown_fraction'] * q_rec_design * 1000   
        self.Qru = self.Er / plant.design['rec_su_delay']
        self.mdot_r_min = plant.design['f_rec_min'] * m_rec_design
        self.mdot_r_max = plant.design['csp_pt_rec_max_oper_frac'] * m_rec_design  
        self.T_rout_min = plant.design['T_rout_min']  
        self.T_rout_max = plant.design['T_rout_max'] 

        # TES parameters
        self.Eu = q_pb_design * plant.design['tshours']  * 1000.
        self.Cp = plant.get_cp_htf(0.5*(plant.design['T_htf_cold_des']+plant.design['T_htf_hot_des'])) * 1.e-3  
        self.mass_cs_min = plant.design['mass_cs_min_frac'] * m_active_cold_max
        self.mass_cs_max = m_active_cold_max
        self.mass_hs_min = plant.design['mass_hs_min_frac'] * m_active_hot_max
        self.mass_hs_max = m_active_hot_max
        self.T_cs_min = plant.design['T_cs_min'] 
        self.T_cs_max = plant.design['T_cs_max']
        self.T_hs_min = plant.design['T_hs_min'] 
        self.T_hs_max = plant.design['T_hs_max']   
        self.T_cs_des = plant.design['T_htf_cold_des']
        self.T_hs_des = plant.design['T_htf_hot_des']
        
        # Cycle parameters
        self.Ec = plant.design['startup_frac'] * q_pb_design * 1000.  
        self.Ew = plant.design['startup_frac_warm'] * q_pb_design * 1000. 
        self.eta_des = plant.design['design_eff']    # TODO: Probably good enough for now. ssc calls the power cycle model at full load and design point ambient T to exactly match full-load performance
        self.Qb = plant.design['q_sby_frac'] * q_pb_design * 1000.     
        self.Qc = self.Ec / ceil(plant.design['startup_time'] / min(self.Delta)) / min(self.Delta)      # TODO: Not clear how to best define this with variable time steps.  Using minimum time step for maximum allowable startup energy rate
        self.Ql = plant.design['cycle_cutoff_frac'] * q_pb_design * 1000. 
        self.Qu = plant.design['cycle_max_frac'] * q_pb_design * 1000.   
        self.kl = m_pb_design * plant.design['cycle_cutoff_frac'] * self.Cp 
        self.ku = m_pb_design * plant.design['cycle_max_frac'] * self.Cp 
        self.Wdot_design = q_pb_design * plant.design['design_eff'] * 1000.  
        self.Wdot_p_max = 25000.  # TODO: Fixing this for now, but probably should base off of design parameters
        self.mdot_c_design = m_pb_design
        self.mdot_c_min = plant.design['cycle_cutoff_frac']*m_pb_design 
        self.mdot_c_max = plant.design['cycle_max_frac']*m_pb_design  
        self.T_cin_design = plant.design['T_htf_hot_des'] 
        self.T_cout_min = plant.design['T_cs_min'] 
        self.T_cout_max = plant.design['T_cs_max']
        self.delta_T_design = plant.design['T_htf_hot_des'] - plant.design['T_htf_cold_des']
        self.delta_T_max = max(abs(plant.design['alpha_b'] * self.delta_T_design), plant.design['T_hs_max'] - plant.design['T_cs_min'])

        if plant.design['pc_config'] == 1: # User-defined cycle
            self.set_linearized_params_from_udpc_inputs(plant)
        else:
            print ('Warning: Dispatch optimization parameters are currently only set up for user-defined power cycle. Defaulting to constant efficiency vs load')
            self.etap = self.eta_des  
            self.Wdotl = self.Ql*self.eta_des  
            self.Wdotu = self.Qu*self.eta_des
    
        self.W_delta_plus = (plant.design['pc_rampup']) * self.Wdotu 
        self.W_delta_minus = (plant.design['pc_rampdown']) * self.Wdotu 
        self.W_v_plus = (plant.design['pc_rampup_vl']) * self.Wdotu 
        self.W_v_minus = (plant.design['pc_rampdown_vl']) * self.Wdotu 
        self.Yu = plant.design['Yu']  
        self.Yd = plant.design['Yd']  
        
        # Parastic loads
        self.Ehs = plant.design['p_start'] * nhel 
        self.Wh_track = plant.design['p_track'] * nhel    
        self.Wh_comm = plant.design['p_comm'] * nhel  
        self.estimate_receiver_pumping_parasitic(plant)  # Sets Lr, Pr
        #self.Lr = plant.design['Lr']      
        #self.Pr = plant.design['Pr']       
        self.Wht_full = plant.design['Wht_fract'] * plant.design['Qrec'] * 1000.
        self.Wht_part = plant.design['Wht_fract_partload'] * plant.design['Qrec'] * 1000.
        self.Lc = plant.design['pb_pump_coef'] * m_pb_design / (q_pb_design * 1000) 
        self.Wb = plant.design['Wb_fract']* plant.design['P_ref']*1000.
        self.Wc = plant.design['Wc_fract']* plant.design['P_ref']*1000.
        
        # Cost parameters
        self.alpha = 1.0  
        self.Crec = plant.design['Crec']   
        self.Crsu = plant.design['Crsu']  
        self.Crhsp = plant.design['Crhsp']     
        self.Cpc = plant.design['Cpc']
        self.Ccsu = plant.design['Ccsu'] 
        self.Cchsp = plant.design['Cchsp']
        self.C_delta_w = plant.design['C_delta_w']
        self.C_v_w  = plant.design['C_v_w']  
        self.Ccsb = plant.design['Ccsb'] 
        
        # Indexing and piecewise-linear indexed parameters
        self.nc = len(plant.design['Pc'])
        self.Pc = plant.design['Pc'] 
        self.Bc = plant.design['Bc']      
        self.nfw = len(plant.design['Pfw'])
        self.Pfw = plant.design['Pfw']    
        self.Bfw = plant.design['Bfw'] 
        
        # Other parameters
        self.alpha_b = plant.design['alpha_b']
        self.alpha_T = plant.design['alpha_T']
        self.alpha_m = plant.design['alpha_m']
        self.beta_b = plant.design['beta_b']
        self.beta_T = plant.design['beta_T']
        self.beta_m = plant.design['beta_m']
        self.beta_mT = plant.design['beta_mT']

        return
        
  
    def estimate_receiver_pumping_parasitic(self, plant, nonheated_length = 0.2):
        m_rec_design = plant.get_receiver_design_mass_flow() #kg/s
        Tavg = 0.5*(plant.design['T_htf_cold_des'] + plant.design['T_htf_hot_des'])
        rho = plant.get_density_htf(Tavg)
        visc = plant.get_visc_htf(Tavg)

        npath = 1
        nperpath = plant.design['N_panels']
        if plant.design['Flow_type'] == 1 or plant.design['Flow_type'] == 2:
            npath = 2
            nperpath = int(plant.design['N_panels']/2)
        elif plant.design['Flow_type'] == 9:
            npath = int(plant.design['N_panels']/2)
            nperpath = 2
            
        ntube = int(pi * plant.design['D_rec']/plant.design['N_panels'] / (plant.design['d_tube_out']*1.e-3))  # Number of tubes per panel
        m_per_tube = m_rec_design / npath / ntube  # kg/s per tube
        tube_id = (plant.design['d_tube_out'] - 2*plant.design['th_tube']) / 1000.  # Tube ID in m
        Ac = 0.25*pi*(tube_id**2)
        vel = m_per_tube / rho / Ac  # HTF velocity
        Re = rho * vel * tube_id / visc
        eD = 4.6e-5 / tube_id
        ff = (-1.737*log(0.269*eD - 2.185/Re*log(0.269*eD+14.5/Re)))**-2
        fd = 4*ff 
        Htot = plant.design['rec_height']* (1+nonheated_length)
        dp = 0.5*fd*rho*(vel**2) * (Htot/tube_id + 4*30 + 2*16) * nperpath  # Frictional pressure drop (Pa) (straight tube, 90deg bends, 45def bends)
        dp += rho * 9.8 * plant.design['h_tower']  # Add pressure drop from pumping up the tower
        if nperpath%2 == 1:   
            dp += rho * 9.8 * Htot  
            
        wdot = dp * m_rec_design / rho / plant.design['eta_pump'] / 1.e6   # Pumping parasitic at design point reciever mass flow rate (MWe)
        
        self.Lr = wdot / plant.design['Qrec'] # MWe / MWt
        self.Pr = wdot * 1000. / m_rec_design  # kWe / kg/s
        
        return

    
    def set_default_grid_limits(self):
        n = len(self.Delta)
        self.Wdotnet = [1.e10 for j in range(n)]
        return
    
    def set_time_weighting_factor(self, wt):
        n = len(self.Delta)
        self.D = [wt**(self.Delta_e[j]) for j in range(n)]
        return
    
    def set_initial_state(self, plant):
        m_des = plant.get_design_storage_mass()

        m_hot = (plant.state['csp_pt_tes_init_hot_htf_percent']/100) * m_des  # Available active mass in hot tank
        m_cold = ((100 - plant.state['csp_pt_tes_init_hot_htf_percent'])/100) * m_des   # Available active mass in cold tank
        cp = plant.get_cp_htf(0.5*(plant.state['T_tank_hot_init']+plant.design['T_htf_cold_des'])) # J/kg/K

        self.T_cs0 = min(max(self.T_cs_min, plant.state['T_tank_cold_init']), self.T_cs_max)
        self.T_hs0 = min(max(self.T_hs_min, plant.state['T_tank_hot_init']), self.T_hs_max)    
        self.s0 = min(self.Eu,  m_hot * cp * (plant.state['T_tank_hot_init'] - plant.design['T_htf_cold_des']) * 1.e-3 / 3600)  # Note s0 is calculated internally in the pyomo dispatch model
        self.mass_cs0 = min(max(self.mass_cs_min, m_cold), self.mass_cs_max)
        self.mass_hs0 = min(max(self.mass_hs_min, m_hot), self.mass_hs_max)
        max_allowable_mass = 0.995*self.Eu*3600 /self.Cp/(self.T_hs0 - self.T_cs_des) + self.mass_hs_min  # Max allowable mass for s0 = Eu from dispatch model s0 calculation in pyomo
        self.mass_hs0 = min(self.mass_hs0, max_allowable_mass)
        self.wdot0 = plant.state['wdot0'] * 1000.          
        
        
        self.yr0 = (plant.state['rec_op_mode_initial'] == 2)
        self.yrsb0 = False      # TODO: no official receiver "standby" mode currently exists in ssc.  Might be able to use the new cold-tank recirculation to define this
        self.yrsu0 = (plant.state['rec_op_mode_initial'] == 1)
        self.yrsd0 = False     # TODO: no official receiver "shutdown" mode currently exists in ssc. 
        self.y0 = (plant.state['pc_op_mode_initial'] == 1) 
        self.ycsb0 = (plant.state['pc_op_mode_initial'] == 2) 
        self.ycsu0 = (plant.state['pc_op_mode_initial'] == 0 or plant.state['pc_op_mode_initial'] == 4) 

        self.drsu0 = plant.state['disp_rec_persist0'] if self.yrsu0 else 0.0   
        self.drsd0 = plant.state['disp_rec_persist0'] if plant.state['rec_op_mode_initial'] == 0 else 0.0 # TODO: defining time in shutdown mode as time "off", will this work in the dispatch model?
        self.Yu0 = plant.state['disp_pc_persist0'] if self.y0 else 0.0
        self.Yd0 = plant.state['disp_pc_off0'] if (not self.y0) else 0.0
        
        
        # Initial startup energy accumulated
        if isnan(plant.state['pc_startup_energy_remain_initial']):  # ssc seems to report nan when startup is completed
            self.ucsu = self.Ec
        else:   
            self.ucsu0 = max(0.0, self.Ec - plant.state['pc_startup_energy_remain_initial']) 
            if self.ucsu0 > (1.0 - 1.e-6)*self.Ec:
                self.ucsu0 = self.Ec
            
        rec_accum_time = max(0.0, self.Drsu - plant.state['rec_startup_time_remain_init'])
        rec_accum_energy = max(0.0, self.Er - plant.state['rec_startup_energy_remain_init']/1000.)
        self.ursu0 = min(rec_accum_energy, rec_accum_time * self.Qru)  # Note, SS receiver model in ssc assumes full available power is used for startup (even if, time requirement is binding)
        if self.ursu0 > (1.0 - 1.e-6)*self.Er:
            self.ursu0 = self.Er

        self.ursd0 = 0.0   #TODO: How can we track accumulated shut-down energy (not modeled in ssc)

        return
    
    # Approximate initial state of receiver "shutdown" variables using yrsd and ursd at last time point accepted from previous dispatch solution
    def set_approximate_shutdown_state_parameters(self, state, ursd = 0.0, yrsd = 0):
        if state['rec_op_mode_initial'] == 3:  # Receiver is off, use ursd, yrsd from previous dispatch solution to define initial properties
            self.yrsd0 = yrsd
            self.ursd0 = ursd
        else:  # Receiver is not off, ignore previous dispatch solution
            self.yrsd0 = 0
            self.ursd0 = 0.0
        return

    # Set dispatch inputs from ssc estimates, S = dictionary containing selected ssc data
    def set_estimates_from_ssc_data(self, design, S, sscstep, require_Qin_nonzero = True):
        n = len(S['Q_thermal'])
        Qin = S['Q_thermal']*1000
        if require_Qin_nonzero:
            Qin = np.maximum(0.0, S['Q_thermal']*1000)
        self.Qin = util.translate_to_variable_timestep(Qin, sscstep, self.Delta)
        ratio = [0 if S['clearsky'][i] < 0.01 else min(1.0, S['beam'][i]/S['clearsky'][i]) for i in range(n)]   # Ratio of actual to clearsky DNI 
        self.F = util.translate_to_variable_timestep(ratio, sscstep, self.Delta)
        self.F = np.minimum(self.F, 1.0)
        self.Q_cls = self.Qin / np.maximum(1.e-6, self.F)

        n = len(self.Delta)
        self.delta_rs = np.zeros(n)
        self.delta_cs = np.zeros(n)
        for t in range(n):
            self.delta_rs[t] = min(1., max(self.Er / max(self.Qin[t]*self.Delta[t], 1.), self.Drsu/self.Delta[t]))
            self.delta_cs[t] = min(1., self.Ec/(self.Qc*self.Delta[t]) )
        
        self.P_field_rec = util.translate_to_variable_timestep((S['P_tower_pump']+S['pparasi'])*1000, sscstep, self.Delta)

        # Set time-series ambient temperature corrections
        if design['pc_config'] == 1: # User-defined cycle
            # TODO: Note that current user-defined cycle neglects ambient T effects
            Tdry = util.translate_to_variable_timestep(S['tdry'], sscstep, self.Delta)
            etamult, wmult = self.get_ambient_T_corrections_from_udpc_inputs(design, Tdry)
     
            #TODO: Make sure these should be efficiency values and not mulipliers
            self.etaamb  = etamult * design['design_eff']
            self.etac = wmult * design['ud_f_W_dot_cool_des']/100.

            
        else:
            # TODO: Need actual ambient temperature correction for cycle efficiency (ssc dispatch model interpolates from a table of points calculated at off-design ambient T, full load, design HTF T)
            self.etaamb = np.ones(n)  
            self.etac = np.ones(n)     
            print ('Dispatch cycle ambient T corrections are currently only set up for a user-defined cycle. Using default values (1.0) at all time points')

        return
    
    def set_day_ahead_schedule(self, signal, penalty_plus, penalty_minus):
        n = len(signal)
        self.num_signal_hours = n
        self.G = signal
        self.Cg_plus = [penalty_plus for j in range(n)]
        self.Cg_minus = [penalty_minus for j in range(n)]
        return 
    

    def set_linearized_params_from_udpc_inputs(self, plant):
        q_pb_design = plant.get_cycle_thermal_rating()  #MWt
        D = util.interpret_user_defined_cycle_data(plant.design['ud_ind_od'])
        eta_adj_pts = [plant.design['ud_ind_od'][p][3]/plant.design['ud_ind_od'][p][4] for p in range(len(plant.design['ud_ind_od'])) ]
        xpts = D['mpts']
        step = xpts[1] - xpts[0]
        
        # Interpolate for cycle performance at specified min/max load points
        fpts = [plant.design['cycle_cutoff_frac'], plant.design['cycle_max_frac']]
        q, eta = [ [] for v in range(2)]
        for j in range(2):
            p = max(0, min(int((fpts[j] - xpts[0]) / step), len(xpts)-2) )  # Find first point in user-defined array of load fractions for interpolation
            i = 3*D['nT'] + D['nm'] + p    # Index of point in full list of udpc points (at design point ambient T)
            eta_adj = eta_adj_pts[i] + (eta_adj_pts[i+1] - eta_adj_pts[i])/step * (fpts[j] - xpts[p])
            eta.append(eta_adj * plant.design['design_eff'])
            q.append(fpts[j]*q_pb_design * 1000.)

        etap = (q[1]*eta[1]-q[0]*eta[0])/(q[1]-q[0])
        b = q[1]*(eta[1] - etap)
        self.etap = etap
        self.Wdotl = b + self.Ql*self.etap
        self.Wdotu = b + self.Qu*self.etap
        return
    
    # Use off-design ambient T performance in user-defined cycle data to interpolate of ambient temperature corrections. Assumes off-design tempertures are specified at a constant interval
    def get_ambient_T_corrections_from_udpc_inputs(self, design, Tamb):
        n = len(Tamb)  # Tamb = set of ambient temperature points for each dispatch time step
        D = util.interpret_user_defined_cycle_data(design['ud_ind_od'])
        
        Tambpts = np.array(D['Tambpts'])
        i0 = 3*D['nT']+3*D['nm']+D['nTamb']  # first index in udpc data corresponding to performance at design point HTF T, and design point mass flow
        npts = D['nTamb']
        etapts = [ design['ud_ind_od'][j][3]/design['ud_ind_od'][j][4] for j in range(i0, i0+npts)]
        wpts = [ design['ud_ind_od'][j][5] for j in range(i0, i0+npts)]
        
        etamult  = np.ones(n)
        wmult = np.ones(n) 
        Tstep = Tambpts[1] - Tambpts[0]
        for j in range(n):
            i = max(0, min( int((Tamb[j] - Tambpts[0]) / Tstep), npts-2) )
            r = (Tamb[j] - Tambpts[i]) / Tstep
            etamult[j] = etapts[i] + (etapts[i+1] - etapts[i])*r
            wmult[j] = wpts[i] + (wpts[i+1] - wpts[i])*r

        return etamult, wmult
    
    def copy_and_format_indexed_inputs(self):
        """
        Return a copy of the dispatch params, first converting all lists and numpy arrays
         into dicts, where each value has a key equal to the index + 1
        e.g.,  [10, 11, 12]  ->  {1: 10, 2: 11, 3: 12}
        """

        newparams = deepcopy(self)
        for k in vars(newparams).keys():
            val = getattr(newparams,k)
            if (type(val) == type([]) or type(val) == type(np.array([1]))):
                newval = {i+1:val[i] for i in range(len(val))}
                setattr(newparams,k,newval)
        return newparams


class DispatchSoln:
    def __init__(self, dispatch_results=None):
        self.objective_value = 0
        self.num_nl_periods = None
        self.t_start = None
        self.s0 = None
               
        # Outputs with descriptive names (note these are repeats of values below)
        self.cycle_on = []                           # Is cycle on? (y)
        self.cycle_standby = []                      # Is cycle in standby? (ycsb)
        self.cycle_startup = []                      # Is cycle starting up? (ycsu)
        self.receiver_on = []                        # Is receiver generating "usable" thermal power? (yr)
        self.receiver_startup = []                   # Is receiver starting up? (yrsu)
        self.receiver_standby = []                   # Is receiver in standby? (yrsb)
        self.receiver_power = []                     # Thermal power delivered by the receiver (kWt)                    
        self.thermal_input_to_cycle = []             # Cycle thermal power utilization (kWt)
        self.electrical_output_from_cycle = []       # Power cycle electricity generation (kWe)
        self.net_electrical_output = []   # Energy sold to grid (kWe)
        self.tes_soc = []

        #Continuous
        self.drsu = []              # Receiver start-up time inventory (hr)
        self.drsd = []              # Receiver shut-down time inventory (hr)
        self.frsu = []              # Fraction of period used for receiver start-up 
        self.frsd = []              # Fraction of period used for receiver shut-down
        self.lr = []                # Salt pumping power to receiver (kWe)
        self.lc = []                # Salt pumping power to SGS (kWe)
        self.lfw = []               # Feed water power to SGS (kWe)
        self.mass_cs = []           # Mass of HTF in cold storage (kg)
        self.mass_hs = []           # Mass of HTF in hot storage (kg)
        self.mdot_c = []            # Mass flow rate of HTF to the cycle [kg/s]
        self.mdot_r_cs = []         # Mass flow rate of HTF to the receiver to cold storage [kg/s]
        self.mdot_r_hs = []         # Mass flow rate of HTF to the receiver to hot storage [kg/s]
        self.s = []                 # TES reserve quantity (kWht)
        self.T_cout = []            # HTF temperature at cycle outlet (C)
        self.T_cs = []              # Cold storage temperature (C)
        self.T_hs = []              # Hot storage temperature (C)
        self.T_rout = []            # Receiver outlet temperature (C)
        self.ucsu = []              # Cycle start-up energy inventory (kWht)
        self.ursu = []              # Receiver start-up energy inventory (kWht)
        self.ursd = []              # Receiver shut-down energy inventory (kWht)
        self.wdot = []              # Power cycle electricity generation (kWe)
        self.wdot_delta_plus = []   # Power cycle ramp-up (kWe)
        self.wdot_delta_minus = []  # Power cycle ramp-down (kWe)
        self.wdot_v_plus = []       # Power cycle ramp-up beyond designated limit (kWe)
        self.wdot_v_minus = []      # Power cycle ramp-up beyond designated limit (kWe)
        self.wdot_s = []            # Energy sold to grid (kWe)
        self.wdot_p = []            # Energy purchased from the grid (kWe)
        self.x = []                 # Cycle thermal power utilization (kWt)
        self.xr = []                # Thermal power delivered by the receiver (kWt)

        #Binary
        self.yr = []        # Is receiver generating "usable" thermal power?
        self.yrsb = []      # Is receiver in standby?
        self.yrsu = []      # Is receiver starting up?
        self.yrsd = []      # Is receiver shutting down? 
        self.yrsup = []     # Is receiver cold start-up penalty incurred (from off)?
        self.yrhsp = []     # Is receiver hot start-up penalty incurred (from standby)?
        self.yrsdp = []     # Is receiver shut-down penalty incurrend?
        self.y = []         # Is cycle generating electric power?
        self.ycsb = []      # Is cycle in standby?
        self.ycsu = []      # Is cycle starting up?
        self.ycsdp = []     # Is cycle shutting down?
        self.ycsup = []     # Is cycle cold start-up penalty incurrent (from off)?
        self.ychsp = []     # Is cycle hot start-up penalty incurrent (from standby)?
        self.ycgb = []      # Did cycle begin electric power generation?
        self.ycge = []      # Did cycle stop electric power generation?
        self.ycsd = []      # Brought over from RTDispatchOutputs
 
        if dispatch_results is not None:
            self.import_dispatch_results(dispatch_results)

        return

    def import_dispatch_results(self, results):
        self.objective_value = pe.value(results.OBJ)
        self.cycle_on = np.array([pe.value(results.y[t]) for t in results.T])
        self.cycle_standby = np.array([pe.value(results.ycsb[t]) for t in results.T])
        self.cycle_startup = np.array([pe.value(results.ycsu[t]) for t in results.T])
        self.receiver_on = np.array([pe.value(results.yr[t]) for t in results.T])
        self.receiver_startup = np.array([pe.value(results.yrsu[t]) for t in results.T])
        self.receiver_standby = np.array([pe.value(results.yrsb[t]) for t in results.T])
        self.receiver_power = np.array([pe.value(results.xr[t]) for t in results.T])
        self.thermal_input_to_cycle = np.zeros_like(self.receiver_power)
        for t in results.T:
            if t in results.T_nl:
                self.thermal_input_to_cycle[t-results.t_start] = pe.value(results.x_calc[t])
            else:
                self.thermal_input_to_cycle[t-results.t_start] = pe.value(results.x[t])
        self.electrical_output_from_cycle = np.array([pe.value(results.wdot[t]) for t in results.T])
        self.net_electrical_output = np.array([pe.value(results.wdot_s[t]) for t in results.T])
        self.tes_soc = np.array([pe.value(results.s[t]) for t in results.T])
        self.num_nl_periods = results.t_transition - results.t_start + 1    # track num_nl_periods
        self.t_start = results.t_start
        #Additional outputs from optimization model (note some repeat from above)
        self.s0 = pe.value(results.s0)
        #Continuous
        self.drsu = np.array([pe.value(results.drsu[t]) for t in results.T])
        self.drsd = np.array([pe.value(results.drsd[t]) for t in results.T])
        self.frsd = np.array([pe.value(results.frsd[t]) for t in results.T])
        self.frsu = np.array([pe.value(results.frsu[t]) for t in results.T])
        self.lr = np.array([pe.value(results.lr[t]) for t in results.T_nl])
        self.lc = np.array([pe.value(results.lc[t]) for t in results.T_nl])
        self.lfw = np.array([pe.value(results.lfw[t]) for t in results.T_nl])
        self.mass_cs = np.array([pe.value(results.mass_cs[t]) for t in results.T_nl])
        self.mass_hs = np.array([pe.value(results.mass_hs[t]) for t in results.T_nl])
        self.mdot_c = np.array([pe.value(results.mdot_c[t]) for t in results.T_nl])
        self.mdot_r_cs = np.array([pe.value(results.mdot_r_cs[t]) for t in results.T_nl])
        self.mdot_r_hs= np.array([pe.value(results.mdot_r_hs[t]) for t in results.T_nl])
        self.s = np.array([pe.value(results.s[t]) for t in results.T_l])
        self.T_cout = np.array([pe.value(results.T_cout[t]) for t in results.T_nl])
        self.T_cs = np.array([pe.value(results.T_cs[t]) for t in results.T_nl])
        self.T_hs = np.array([pe.value(results.T_hs[t]) for t in results.T_nl])
        self.T_rout = np.array([pe.value(results.T_rout[t]) for t in results.T_nl])
        self.ucsu = np.array([pe.value(results.ucsu[t]) for t in results.T])
        #self.ucsd = np.array([pe.value(results.ucsd[t]) for t in results.T])
        self.ursu = np.array([pe.value(results.ursu[t]) for t in results.T])
        self.ursd = np.array([pe.value(results.ursd[t]) for t in results.T])
        self.wdot = np.array([pe.value(results.wdot[t]) for t in results.T])
        self.wdot_delta_plus = np.array([pe.value(results.wdot_delta_plus[t]) for t in results.T])
        self.wdot_delta_minus = np.array([pe.value(results.wdot_delta_minus[t]) for t in results.T])
        self.wdot_v_plus = np.array([pe.value(results.wdot_v_plus[t]) for t in results.T])
        self.wdot_v_minus = np.array([pe.value(results.wdot_v_minus[t]) for t in results.T])
        self.wdot_s = np.array([pe.value(results.wdot_s[t]) for t in results.T])
        self.wdot_p = np.array([pe.value(results.wdot_p[t]) for t in results.T])
        self.x = np.array([pe.value(results.x[t]) for t in results.T_l])
        self.xr = np.array([pe.value(results.xr[t]) for t in results.T])
        #self.xrsu = np.array([pe.value(results.xrsu[t]) for t in results.T])
        #Binary
        self.yr = np.array([pe.value(results.yr[t]) for t in results.T])
        self.yrhsp = np.array([pe.value(results.yrhsp[t]) for t in results.T])
        self.yrsb = np.array([pe.value(results.yrsb[t]) for t in results.T])
        self.yrsd = np.array([pe.value(results.yrsd[t]) for t in results.T])
        self.yrsdp = np.array([pe.value(results.yrsdp[t]) for t in results.T])
        self.yrsu = np.array([pe.value(results.yrsu[t]) for t in results.T])
        self.yrsup = np.array([pe.value(results.yrsup[t]) for t in results.T])
        self.y = np.array([pe.value(results.y[t]) for t in results.T])
        self.ychsp = np.array([pe.value(results.ychsp[t]) for t in results.T])
        self.ycsb = np.array([pe.value(results.ycsb[t]) for t in results.T])
        self.ycsd = np.array([pe.value(results.ycsd[t]) for t in results.T])
        self.ycsdp = np.array([pe.value(results.ycsdp[t]) for t in results.T])
        self.ycsu = np.array([pe.value(results.ycsu[t]) for t in results.T])
        self.ycsup = np.array([pe.value(results.ycsup[t]) for t in results.T])
        self.ycgb = np.array([pe.value(results.ycgb[t]) for t in results.T])
        self.ycge = np.array([pe.value(results.ycge[t]) for t in results.T])
  
    
    def get_solution_at_ssc_steps(self, disp_params, sscstep, horizon):
        inds = np.where(np.array(disp_params.Delta_e) <= horizon+0.0001)[0]
        dt = [disp_params.Delta[i] for i in inds]
        R = {}
        for k in vars(self):
            vals = getattr(self, k)
            if (type(vals) == type([]) or type(vals) == type(np.array(1))) and len(vals) >= len(inds):
                vals = [vals[i] for i in inds]
                R[k] = util.translate_to_fixed_timestep(vals, dt, sscstep)  
        return R 
    
    def get_value_at_time(self, disp_params, time, name):
        if len(self.y) == 0:
            return False
        i = np.where(np.array(disp_params.Delta_e) <= time+0.0001)[0][-1] 
        vals = getattr(self, name)
        return vals[i]


class DispatchTargets:
    def __init__(self, dispatch_soln=None, plant=None, dispatch_params=None, sscstep=None, horizon=None):
        self.q_pc_target_su_in = []         # Target thermal power to cycle for startup (MWt)
        self.q_pc_target_on_in = []         # Target thermal power to cycle for operation (MWt)
        self.q_pc_max_in = []               # Max thermal power to cycle (MWt)
        self.is_rec_su_allowed_in = []      # Is receiver startup/operation allowed
        self.is_rec_sb_allowed_in = []      # Is receiver standby allowed? 
        self.is_pc_su_allowed_in = []       # Is power cycle startup/operation allowed?
        self.is_pc_sb_allowed_in = []       # Is power cycle standby allowed?

        #TODO: Any additional targets from new dispatch model?

        if dispatch_soln is not None:
            self.set_from_dispatch_solution(dispatch_soln, plant, dispatch_params, sscstep/3600., horizon)

        return

    def asdict(self):
        return {
            'q_pc_target_su_in': self.q_pc_target_su_in,
            'q_pc_target_on_in': self.q_pc_target_on_in,
            'q_pc_max_in': self.q_pc_max_in,
            'is_rec_su_allowed_in': self.is_rec_su_allowed_in,
            'is_rec_sb_allowed_in': self.is_rec_sb_allowed_in,
            'is_pc_su_allowed_in': self.is_pc_su_allowed_in,
            'is_pc_sb_allowed_in': self.is_pc_sb_allowed_in,
            }

    def set_from_dispatch_solution(self, disp_soln, plant, disp_params, sscstep, horizon):
        """
        Translate to or generate SSC model inputs from select dispatch model outputs

        Inputs:     disp_soln
        Outputs:    setting object member variables 'is_rec_su_allowed_in', etc.

        TODO
        - extract this line and the above two parameters as they don't deal with the dispatch solution
            D['q_pc_max_in'] = [q_pc_max_val for t in range(n)]

        - move 'Set binary inputs' to a subsequent loop
        """
        n = len(disp_soln.cycle_on)  # Number of time periods in full dispatch solution (variable time steps)
        dt = disp_params.Delta

        q_pc_max_val = plant.get_cycle_thermal_rating() * plant.design['cycle_max_frac']  # Maximum cycle thermal input from design parameters (MWt)

        is_simple_receiver = True if len(disp_soln.receiver_on) == 0 else False
        
        y = disp_soln.cycle_on
        ycsu = disp_soln.cycle_startup
        ycsb = disp_soln.cycle_standby
        q_pc_target = disp_soln.thermal_input_to_cycle
        
        if not is_simple_receiver:
            yr = disp_soln.receiver_on
            yrsu = disp_soln.receiver_startup
            yrsb = disp_soln.receiver_standby
        

        D = {}
        if not is_simple_receiver:
            D['is_rec_su_allowed_in'] = [ 1 if (yr[t] + yrsu[t] + yrsb[t]) > 0.001 else 0 for t in range(n)]  # Receiver on, startup, or standby
            D['is_rec_sb_allowed_in'] = [ 1 if yrsb[t] > 0.001 else 0 for t in range (n)]  # Receiver standby
        else:
            D['is_rec_su_allowed_in'] = [ 1 for t in range(n)]  
            D['is_rec_sb_allowed_in'] = [ 0 for t in range(n)]  
        
        D['is_pc_su_allowed_in'] = [ 1 if (y[t] + ycsu[t]) > 0.001 else 0 for t in range(n)]  # Cycle on or startup
        D['is_pc_sb_allowed_in'] = [ 1 if ycsb[t] > 0.001 else 0 for t in range(n)]  # Cyle standby

        #TODO: Might need to modify q_pc_target_on_in and q_pc_max_in for timesteps split between cycle startup and operation (e.g. 1383 - 1414 of csp_solver_core.cpp in mjwagner2/ssc/daotk-develop)
        D['q_pc_target_su_in'] = [disp_params.Qc/1000. if ycsu[t] > 0.001 else 0.0 for t in range(n)]
        D['q_pc_target_on_in'] = [q_pc_target[t]/1000. for t in range(n)]
        D['q_pc_max_in'] = [q_pc_max_val for t in range(n)]

        # Translate to fixed ssc timestep and limit arrays to the desired time horizon (length of dispatch target input arrays in ssc needs to match the designated simulation time))
        npts = int(horizon / sscstep)
        for k in D.keys():
            vals = util.translate_to_fixed_timestep(D[k], dt, sscstep)  # Translate from variable dispatch timestep to fixed ssc time step
            vals = [vals[j] for j in range(npts)]                     # Truncate to only the points needed for the ssc solution
            if k in ['is_rec_su_allowed_in', 'is_rec_sb_allowed_in', 'is_pc_su_allowed_in', 'is_pc_sb_allowed_in']:  # Set binary inputs
                vals = [1 if v > 0.001 else 0 for v in vals]
            setattr(self, k, vals)

        return



class DispatchWrap:
    def __init__(self, plant, params):

        ## DISPATCH INPUTS ###############################################################################################################################
        # Input data files: weather, masslow, clearsky DNI must have length of full annual array based on ssc time step size
        #--- Simulation start point and duration
        self.start_date = None
        self.sim_days = None
        # self.plant = None                             # Plant design and operating properties
        
        self.user_defined_cycle_input_file = '../../librtdispatch/udpc_noTamb_dependency.csv'  # Only required if cycle_type is user_defined
        self.ground_truth_weather_file = './model-validation/input_files/weather_files/ssc_weatherfile_1min_2018.csv'    # Weather file derived from CD data: DNI, ambient temperature,
                                                                                                                         #  wind speed, etc. are averaged over 4 CD weather stations,
                                                                                                                         #  after filtering DNI readings for bad measurements. 

        ## DISPATCH PERSISTING INTERMEDIARIES ############################################################################################################
        self.first_run = True                           # Is this the first time run() is called?
        self.ursd_last = 0
        self.yrsd_last = 0
        self.dispatch_params = DispatchParams()         # Structure to contain all inputs for dispatch model 
        self.current_time = 0                           # Current time (tracked in standard time, not local time)
        self.is_initialized = False                     # Has solution already been initalized?        
        self.CD_data_for_plotting = {}                  # Only used if control_cycle = 'CD_data' 


        ## DISPATCH OUTPUTS FOR INPUT INTO SSC ###########################################################################################################
        # see: ssc_dispatch_targets, which is a dispatch.DispatchTargets object


        ## SSC OUTPUTS ###################################################################################################################################
        self.results = None                             # Results

        self.weather_data_for_dispatch = []
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

        params['clearsky_data'] = util.get_clearsky_data(params['clearsky_file'], params['time_steps_per_hour'])   # legacy call. TODO: cleanup?
        params['solar_resource_data'] = util.get_ground_truth_weather_data(self.ground_truth_weather_file, params['time_steps_per_hour'])

        self.plant = plant                              # Plant design and operating properties
        self.params = params

        for key,value in params.items():
            setattr(self, key, value)

        # Aliases (that could be combined and removed)
        self.start_date = datetime.datetime(self.start_date_year, 1, 1, 8, 0, 0)      # needed for schedules TODO: fix spanning years (note the 8)
        self.ssc_time_steps_per_hour = params['time_steps_per_hour']
        self.use_transient_model = params['is_rec_model_trans']
        self.use_transient_startup = params['is_rec_startup_trans']
        self.price_data = params['dispatch_factors_ts']
        self.ground_truth_weather_data = params['solar_resource_data']

        # Initialize and adjust above parameters
        self.initialize()

        return


    def initialize(self):
        
        # Check combinations of control conditions
        if self.is_optimize and (self.control_field == 'CD_data' or self.control_receiver == 'CD_data'):
            print ('Warning: Dispatch optimization is being used with field or receiver operation derived from CD data. Receiver can only operate when original CD receiver was operating')
        if self.control_receiver == 'CD_data' and self.control_field != 'CD_data':
            print ('Warning: Receiver flow is controlled from CD data, but field tracking fraction is controlled by ssc. Temperatures will likely be unrealistically high')


        self.weather_data_for_dispatch = util.create_empty_weather_data(self.ground_truth_weather_data, self.ssc_time_steps_per_hour)
        self.current_day_schedule = np.zeros(24*self.day_ahead_schedule_steps_per_hour)
        self.next_day_schedule = np.zeros(24*self.day_ahead_schedule_steps_per_hour)
        if int(util.get_time_of_day(self.start_date)) == 0 and self.use_day_ahead_schedule and self.day_ahead_schedule_from == 'calculated':
            self.schedules.append(None)
        
        # TODO: Fix spanning years. The 'date' must be of the same year as the weather data, and not at the very end of year
        self.current_forecast_weather_data = DispatchWrap.update_forecast_weather_data(
                date=self.start_date - datetime.timedelta(hours = 24-self.forecast_issue_time),
                current_forecast_weather_data=util.create_empty_weather_data(self.ground_truth_weather_data, self.ssc_time_steps_per_hour),
                ssc_time_steps_per_hour=self.ssc_time_steps_per_hour,
                forecast_steps_per_hour=self.forecast_steps_per_hour,
                ground_truth_weather_data=self.ground_truth_weather_data,
                forecast_issue_time=self.forecast_issue_time,
                day_ahead_schedule_time=self.day_ahead_schedule_time,
                clearsky_data=self.clearsky_data
                )


        self.is_initialized = True
        return


    #--- Run simulation
    def run(self, start_date, timestep_days, horizon, retvars, ursd_last, yrsd_last, current_forecast_weather_data, weather_data_for_dispatch,
            schedules, current_day_schedule, next_day_schedule, f_estimates_for_dispatch_model, initial_plant_state=None):
        """horizon = [s]"""
        if self.first_run == True:
            if ursd_last is None: ursd_last = self.ursd_last
            if yrsd_last is None: yrsd_last = self.yrsd_last
            if weather_data_for_dispatch is None: weather_data_for_dispatch = self.weather_data_for_dispatch
            if current_day_schedule is None: current_day_schedule = self.current_day_schedule
            if next_day_schedule is None: next_day_schedule = self.next_day_schedule
            if schedules is None: schedules = self.schedules
            if current_forecast_weather_data is None: current_forecast_weather_data = self.current_forecast_weather_data

            self.first_run == False

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
        #TODO: don't pass full copy of plant object to dispatch_wrap, just it's state, etc.
        D = self.plant.design.copy()
        D.update(self.plant.get_state())
        D.update(self.plant.flux_maps)
        D['time_start'] = int(util.get_time_of_year(self.start_date))
        D['time_stop'] = int(util.get_time_of_year(self.start_date) + self.sim_days*24*3600)
        reupdate_ssc_constants(D, self.params)

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

        #--- Update "forecasted" weather data (if relevant)
        if self.is_optimize and (tod == self.forecast_issue_time*3600):
            self.current_forecast_weather_data = DispatchWrap.update_forecast_weather_data(
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

            #--- Update weather to use in dispatch optimization for this optimization horizon
            self.weather_data_for_dispatch = update_dispatch_weather_data(
                weather_data = self.weather_data_for_dispatch,
                replacement_real_weather_data = self.ground_truth_weather_data,
                replacement_forecast_weather_data = self.current_forecast_weather_data,
                datetime = time,
                total_horizon = horizon/3600.,      # [s]
                dispatch_horizon = self.dispatch_weather_horizon
                )

            #--- Run ssc for dispatch estimates: (using weather forecast time resolution for weather data and specified ssc time step)
            npts_horizon = int(horizon/3600 * nph)
            R_est = f_estimates_for_dispatch_model(
                plant_design = D,
                toy = toy,
                horizon = horizon,      # [s]
                weather_data = self.weather_data_for_dispatch,
                N_pts_horizon = npts_horizon,
                clearsky_data = self.clearsky_data,
                start_pt = startpt
            )

            #--- Set dispatch optimization properties for this time horizon using ssc estimates
            disp_in = setup_dispatch_model(
                R_est = R_est,
                freq = freq,
                horizon = horizon,      # [s]
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
                startpt = startpt,
                toy = toy,
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
                
            dispatch_soln = run_dispatch_model(disp_in, include)

            # Populate results
            if self.is_optimize:
                retvars += vars(DispatchTargets()).keys()
            if dispatch_soln is not None:
                Rdisp_all = dispatch_soln.get_solution_at_ssc_steps(self.dispatch_params, sscstep/3600., freq/3600.)
                Rdisp = {'disp_'+key:value for key,value in Rdisp_all.items() if key in retvars}

                # TODO: make the time triggering more robust; shouldn't be an '==' as the program may be offline at the time or running at intervals
                #  that won't exactly hit it
                if self.use_day_ahead_schedule and self.day_ahead_schedule_from == 'calculated' and tod/3600 == self.day_ahead_schedule_time:
                    self.next_day_schedule = get_day_ahead_schedule(
                        day_ahead_schedule_steps_per_hour = self.day_ahead_schedule_steps_per_hour,
                        Delta = self.dispatch_params.Delta,
                        Delta_e = self.dispatch_params.Delta_e,
                        net_electrical_output = dispatch_soln.net_electrical_output,
                        day_ahead_schedule_time = self.day_ahead_schedule_time
                        )

                    weather_at_day_ahead_schedule = get_weather_at_day_ahead_schedule(self.weather_data_for_dispatch, startpt, npts_horizon)
                    self.weather_at_schedule.append(weather_at_day_ahead_schedule)  # Store weather used at the point in time the day ahead schedule was generated

                #--- Set ssc dispatch targets
                ssc_dispatch_targets = DispatchTargets(dispatch_soln, self.plant, self.dispatch_params, sscstep, freq/3600.)

                #--- Save these values for next estimates
                ursd_last = dispatch_soln.get_value_at_time(self.dispatch_params, freq/3600, 'ursd')      # set to False when it doesn't exists 
                yrsd_last = dispatch_soln.get_value_at_time(self.dispatch_params, freq/3600, 'yrsd')      # set to False when it doesn't exists

            else:  # Infeasible solution was returned, revert back to running ssc without dispatch targets
                Rdisp = None
                ssc_dispatch_targets = None

        tod = int(util.get_time_of_day(self.start_date))
        if tod == 0 and self.use_day_ahead_schedule and self.day_ahead_schedule_from == 'calculated':
            self.current_day_schedule = [s for s in self.next_day_schedule]
            self.schedules.append(self.current_day_schedule)

        dispatch_outputs = {
            'ssc_dispatch_targets': ssc_dispatch_targets,
            'Rdisp': Rdisp,
            'ursd_last': ursd_last,
            'yrsd_last': yrsd_last,
            'current_forecast_weather_data': self.current_forecast_weather_data,
	        'weather_data_for_dispatch': self.weather_data_for_dispatch,
            'schedules': self.schedules,
	        'current_day_schedule': self.current_day_schedule,
	        'next_day_schedule': self.next_day_schedule
        }

        # Read NVE schedules (if not already read during rolling horizon calculations)
        if self.is_optimize == False and self.use_day_ahead_schedule and self.day_ahead_schedule_from == 'NVE':
            for j in range(self.sim_days):
                date = datetime.datetime(self.start_date.year, self.start_date.month, self.start_date.day + j)
                dispatch_outputs['schedules'].append(self.get_CD_NVE_day_ahead_schedule(date))

        return dispatch_outputs


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


def reupdate_ssc_constants(D, params):
    D['solar_resource_data'] = params['solar_resource_data']
    D['dispatch_factors_ts'] = params['dispatch_factors_ts']

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
        D['rec_clearsky_dni'] = params['clearsky_data'].tolist()
    elif params['control_receiver'] == 'ssc_actual_dni':
        D['rec_clearsky_fraction'] = params['rec_clearsky_fraction']

    return




def update_dispatch_weather_data(weather_data, replacement_real_weather_data, replacement_forecast_weather_data, datetime, total_horizon, dispatch_horizon):
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


def setup_dispatch_model(R_est, freq, horizon, include_day_ahead_in_dispatch,
    dispatch_params, plant, nonlinear_model_time, use_linear_dispatch_at_night,
    clearsky_data, night_clearky_cutoff, dispatch_steplength_array, dispatch_steplength_end_time,
    disp_time_weighting, price, sscstep, avg_price, avg_price_disp_storage_incentive,
    avg_purchase_price, day_ahead_tol_plus, day_ahead_tol_minus, startpt, toy,
    tod, current_day_schedule, day_ahead_pen_plus, day_ahead_pen_minus,
    dispatch_horizon, night_clearsky_cutoff, ursd_last, yrsd_last):
    """horizon = [s]"""

    #--- Set dispatch optimization properties for this time horizon using ssc estimates
    ##########
    ##  There's already a lot of the dispatch_params member variables set here, which set_initial_state draws from
    ##########
    # Initialize dispatch model inputs
    dispatch_params.set_dispatch_time_arrays(dispatch_steplength_array, dispatch_steplength_end_time,
        dispatch_horizon, nonlinear_model_time, disp_time_weighting)
    dispatch_params.set_fixed_parameters_from_plant_design(plant)
    dispatch_params.set_default_grid_limits()
    dispatch_params.disp_time_weighting = disp_time_weighting
    dispatch_params.set_initial_state(plant)  # Set initial plant state for dispatch model
    
    # Update approximate receiver shutdown state from previous dispatch solution (not returned from ssc)
    dispatch_params.set_approximate_shutdown_state_parameters(plant.state, ursd = ursd_last, yrsd = yrsd_last)  # Set initial state parameters related to shutdown from dispatch model (because this cannot be derived from ssc)

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

    dispatch_params.set_estimates_from_ssc_data(plant.design, R_est, sscstep/3600.) 
    
    
    #--- Set day-ahead schedule in dispatch parameters
    if include_day_ahead_in_dispatch:  
        day_ahead_horizon = 24 - int(tod/3600)   # Number of hours of day-ahead schedule to use.  This probably only works in the dispatch model if time horizons are updated at integer multiples of an hour
        use_schedule = [current_day_schedule[s]*1000 for s in range(24-day_ahead_horizon, 24)]   # kWhe
        dispatch_params.set_day_ahead_schedule(use_schedule, day_ahead_pen_plus/1000, day_ahead_pen_minus/1000)
        

    #--- Create copy of params, and convert all lists and numpy arrays into dicts, where each value has a key equal to the index + 1
    disp_in = dispatch_params.copy_and_format_indexed_inputs()     # dispatch.DispatchParams object

    return disp_in


def run_dispatch_model(disp_in, include, transition=0):
    disp_in.transition = transition
    rt = dispatch_model.RealTimeDispatchModel(disp_in, include)
    rt_results = rt.solveModel()
    
    if rt_results.solver.termination_condition == TerminationCondition.infeasible:
        return False

    return DispatchSoln(rt.model)


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


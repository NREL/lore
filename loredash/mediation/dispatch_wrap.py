import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from math import ceil, pi, log, isnan
import numpy as np
from copy import deepcopy
import pyomo.environ as pe
from pyomo.opt import TerminationCondition
import datetime

from mediation import dispatch_model

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
        self.avg_price_disp_storage_incentive = 0  #  Average electricity price ($/kWh) used in dispatch model storage inventory incentive ($/kWh).  Note, high values cause the solutions to max out storage rather than generate electricity

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

        # Initial setup for part-load cycle efficiency (this will be over-written using power cycle off-design output tables from ssc)
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
    def set_estimates_from_ssc_data(self, plant, S, sscstep, require_Qin_nonzero = True):
        n = len(S['Q_thermal'])

        Qin = np.array([S['Q_thermal'][i]*1000. for i in range(n)])   #kWt
        if require_Qin_nonzero:
            Qin = np.maximum(0.0, Qin)
        self.Qin = translate_to_variable_timestep(Qin, sscstep, self.Delta)

        field_receiver_parasitic = [(S['P_tower_pump'][i]+S['pparasi'][i])*1000. for i in range(n)]  # kWe
        self.P_field_rec = translate_to_variable_timestep(field_receiver_parasitic, sscstep, self.Delta)

        clearsky_adjusted = np.maximum(S['beam'], S['clearsky'])  
        ratio = [0 if clearsky_adjusted[i] < 0.01 else min(1.0, S['beam'][i]/clearsky_adjusted[i]) for i in range(n)]   # Ratio of actual to clearsky DNI 
        self.F = translate_to_variable_timestep(ratio, sscstep, self.Delta)
        self.F = np.minimum(self.F, 1.0)
        self.Q_cls = self.Qin / np.maximum(1.e-6, self.F)

        n = len(self.Delta)
        self.delta_rs = np.zeros(n)
        self.delta_cs = np.zeros(n)
        for t in range(n):
            self.delta_rs[t] = min(1., max(self.Er / max(self.Qin[t]*self.Delta[t], 1.), self.Drsu/self.Delta[t]))
            self.delta_cs[t] = min(1., self.Ec/(self.Qc*self.Delta[t]) )
    
        # Set time-series power cycle ambient temperature corrections and cycle part-load efficiency
        Tdb = translate_to_variable_timestep(S['tdry'], sscstep, self.Delta)
        self.set_off_design_cycle_inputs(plant, Tdb, S)

        return
    
    def set_day_ahead_schedule(self, signal, penalty_plus, penalty_minus):
        n = len(signal)
        self.num_signal_hours = n
        self.G = signal
        self.Cg_plus = [penalty_plus for j in range(n)]
        self.Cg_minus = [penalty_minus for j in range(n)]
        return 
    
    # Set parameters in dispatch model for off-design cycle performance
    def set_off_design_cycle_inputs(self, plant, Tdb, ssc_outputs):
        is_ssc_tables = 'cycle_eff_load_table' in ssc_outputs and 'cycle_eff_Tdb_table' in ssc_outputs and 'cycle_wcond_Tdb_table' in ssc_outputs
    
        #--- Cycle part-load efficiency (not a function of time, but setting here to use outputs from ssc estimates)
        if is_ssc_tables:
            q_pb_design = plant.get_cycle_thermal_rating()
            nload = len(ssc_outputs['cycle_eff_load_table'])
            xpts = [ssc_outputs['cycle_eff_load_table'][i][0]/q_pb_design for i in range(nload)]    # Load fraction
            etapts = [ssc_outputs['cycle_eff_load_table'][i][1] for i in range(nload)]              # Efficiency 
            self.set_linearized_cycle_part_load_params(plant, xpts, etapts)
        elif plant.design['pc_config'] == 1:    # Tables not returned from ssc, but can be taken from user-defined cycle inputs
            D = interpret_user_defined_cycle_data(plant.design['ud_ind_od'])
            k = 3*D['nT'] + D['nm']
            xpts = D['mpts']   # Load fraction
            etapts = [plant.design['design_eff'] * (plant.design['ud_ind_od'][k+p][3]/plant.design['ud_ind_od'][k+p][4]) for p in range(len(xpts))]  # Efficiency
            self.set_linearized_cycle_part_load_params(plant, xpts, etapts)
        else:
            print ('WARNING: Dispatch optimization cycle part-load efficiency is not set up. Defaulting to constant efficiency vs load')
            self.etap = self.eta_des                         
            self.Wdotl = self.Ql*self.eta_des  
            self.Wdotu = self.Qu*self.eta_des

        #--- Cycle ambient-temperature efficiency corrections
        if is_ssc_tables:
            nT = len(ssc_outputs['cycle_eff_Tdb_table'])
            Tpts = [ssc_outputs['cycle_eff_Tdb_table'][i][0] for i in range(nT)]                                
            etapts = [ssc_outputs['cycle_eff_Tdb_table'][i][1]* plant.design['design_eff'] for i in range(nT)]  # Efficiency
            wcondfpts = [ssc_outputs['cycle_wcond_Tdb_table'][i][1] for i in range(nT)]                         # Fraction of cycle design gross output consumed by cooling
            self.set_cycle_ambient_corrections(Tdb, Tpts, etapts, wcondfpts)
        elif plant.design['pc_config'] == 1:          # Tables not returned from ssc, but can be taken from user-defined cycle inputs
            D = interpret_user_defined_cycle_data(plant.design['ud_ind_od'])
            k = 3*D['nT']+3*D['nm']+D['nTamb']  # first index in udpc data corresponding to performance at design point HTF T, and design point mass flow
            npts = D['nTamb']
            etapts = [ plant.design['design_eff'] * (plant.design['ud_ind_od'][j][3]/plant.design['ud_ind_od'][j][4]) for j in range(k, k+npts)]  # Efficiency
            wcondfpts = [(plant.design['ud_f_W_dot_cool_des']/100.)*plant.design['ud_ind_od'][j][5] for j in range(k, k+npts)]                    # Fraction of cycle design gross output consumed by cooling
            self.set_cycle_ambient_corrections(Tdb, D['Tambpts'], etapts, wcondfpts)
        else:
            print ('WARNING: Dispatch optimization cycle ambient T corrections are not set up. Using default values (1.0) at all time points')
            n = len(Tdb)
            self.etaamb = np.ones(n) * plant.design['design_eff']  # Design point efficiency at all ambient T
            self.etac = np.zeros(n)                                # No condenser parasitic requirement
        return
        

    def set_linearized_cycle_part_load_params(self, plant, xfpts, etapts):
        q_pb_design = plant.get_cycle_thermal_rating()
        fpts = [plant.design['cycle_cutoff_frac'], plant.design['cycle_max_frac']]
        step = xfpts[1] - xfpts[0]
        q, eta = [ [] for v in range(2)]
        for j in range(2):
            p = max(0, min(int((fpts[j] - xfpts[0]) / step), len(xfpts)-2) )               # Find first point in user-defined array of load fractions
            eta.append(etapts[p] + (etapts[p+1] - etapts[p])/step * (fpts[j] - xfpts[p]))  
            q.append(fpts[j]*q_pb_design * 1000.)  # kW
        etap = (q[1]*eta[1]-q[0]*eta[0])/(q[1]-q[0])
        b = q[1]*(eta[1] - etap)
        self.etap = etap
        self.Wdotl = b + self.Ql*self.etap
        self.Wdotu = b + self.Qu*self.etap
        return

    def set_cycle_ambient_corrections(self, Tdb, Tpts, etapts, wcondfpts):
        n = len(Tdb)            # Tdb = set of ambient temperature points for each dispatch time step
        npts = len(Tpts)        # Tpts = ambient temperature points with tabulated values
        self.etaamb = np.ones(n)
        self.etac = np.zeros(n) 
        Tstep = Tpts[1] - Tpts[0]
        for j in range(n):
            i = max(0, min( int((Tdb[j] - Tpts[0]) / Tstep), npts-2) )
            r = (Tdb[j] - Tpts[i]) / Tstep
            self.etaamb[j] = etapts[i] + (etapts[i+1] - etapts[i])*r
            self.etac[j] = wcondfpts[i] + (wcondfpts[i+1] - wcondfpts[i])*r
        return

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
        self.net_electrical_output = []              # Energy sold to grid (kWe)
        self.tes_soc = []

        # Outputs that very by nonlinear vs. linear model
        self.receiver_outlet_temp = []               # Receiver outlet temp (C)
        self.cycle_outlet_temp = []                  # Power cycle outlet temp (C)
        self.hot_salt_tank_temp = []                 # Hot salt tank temp (C)
        self.cold_salt_tank_temp = []                # Cold salt tank temp (C)

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
                self.thermal_input_to_cycle[t] = pe.value(
                    results.Cp * results.mdot_c[t] * (
                            results.T_hs[t] - results.T_cout[t])
                )   #  (units.kJ/(degK*kg)) * (kg/s) * (degK) = kJ/s = kW --> no multiple
            else:
                self.thermal_input_to_cycle[t-results.t_start] = pe.value(results.x[t])
        self.electrical_output_from_cycle = np.array([pe.value(results.wdot[t]) for t in results.T])
        self.net_electrical_output = np.array([pe.value(results.wdot_s[t]) for t in results.T])
        self.tes_soc = np.array([pe.value(results.s[t]) for t in results.T])
        self.receiver_outlet_temp = np.zeros_like(self.receiver_power)
        self.cycle_outlet_temp = np.zeros_like(self.receiver_power)
        self.hot_salt_tank_temp = np.zeros_like(self.receiver_power)
        self.cold_salt_tank_temp = np.zeros_like(self.receiver_power)
        for t in results.T:
            if t in results.T_nl:
                self.receiver_outlet_temp[t-results.t_start] = pe.value(results.T_rout[t])
                self.cycle_outlet_temp[t-results.t_start] = pe.value(results.T_cout[t])
                self.hot_salt_tank_temp[t-results.t_start] = pe.value(results.T_hs[t])
                self.cold_salt_tank_temp[t-results.t_start] = pe.value(results.T_cs[t])
            else:
                self.cold_salt_tank_temp[t - results.t_start] = pe.value(results.T_cs_des)
                if results.t_transition == 0:
                    self.receiver_outlet_temp[t - results.t_start] = pe.value(results.T_hs_des)
                    self.cycle_outlet_temp[t - results.t_start] = pe.value(results.T_hs_des)
                    self.hot_salt_tank_temp[t - results.t_start] = pe.value(results.T_hs_des)
                else:
                    self.receiver_outlet_temp[t - results.t_start] = pe.value(results.T_hs[results.t_transition])
                    self.cycle_outlet_temp[t - results.t_start] = pe.value(results.T_hs[results.t_transition])
                    self.hot_salt_tank_temp[t - results.t_start] = pe.value(results.T_hs[results.t_transition])
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
                R[k] = translate_to_fixed_timestep(vals, dt, sscstep)  
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

    def asdict(self,use_lists=False):
        if use_lists:
            return {
                'q_pc_target_su_in':    list(self.q_pc_target_su_in),
                'q_pc_target_on_in':    list(self.q_pc_target_on_in),
                'q_pc_max_in':          list(self.q_pc_max_in),
                'is_rec_su_allowed_in': list(self.is_rec_su_allowed_in),
                'is_rec_sb_allowed_in': list(self.is_rec_sb_allowed_in),
                'is_pc_su_allowed_in':  list(self.is_pc_su_allowed_in),
                'is_pc_sb_allowed_in':  list(self.is_pc_sb_allowed_in),
            }
        return {
            'q_pc_target_su_in':        self.q_pc_target_su_in,
            'q_pc_target_on_in':        self.q_pc_target_on_in,
            'q_pc_max_in':              self.q_pc_max_in,
            'is_rec_su_allowed_in':     self.is_rec_su_allowed_in,
            'is_rec_sb_allowed_in':     self.is_rec_sb_allowed_in,
            'is_pc_su_allowed_in':      self.is_pc_su_allowed_in,
            'is_pc_sb_allowed_in':      self.is_pc_sb_allowed_in,
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
            vals = translate_to_fixed_timestep(D[k], dt, sscstep)  # Translate from variable dispatch timestep to fixed ssc time step
            vals = [vals[j] for j in range(npts)]                     # Truncate to only the points needed for the ssc solution
            if k in ['is_rec_su_allowed_in', 'is_rec_sb_allowed_in', 'is_pc_su_allowed_in', 'is_pc_sb_allowed_in']:  # Set binary inputs
                vals = [1 if v > 0.001 else 0 for v in vals]
            setattr(self, k, vals)

        return

    def update_from_dict(self,target_dict):
        """
        updates attributes of the object using a dictionary as input.  called after data validation takes place for
        the converted object (using asdict).

        parameters
        ============
        target_dict : Dict(str, list) | dictionary containing (validated) dispatch targets

        returns
        ===========
        None (updates attributes of object
        """
        self.q_pc_target_su_in = target_dict['q_pc_target_su_in']
        self.q_pc_target_on_in = target_dict['q_pc_target_on_in']
        self.q_pc_max_in = target_dict['q_pc_max_in']
        self.is_rec_su_allowed_in = target_dict['is_rec_su_allowed_in']
        self.is_rec_sb_allowed_in = target_dict['is_rec_sb_allowed_in']
        self.is_pc_su_allowed_in = target_dict['is_pc_su_allowed_in']
        self.is_pc_sb_allowed_in = target_dict['is_pc_sb_allowed_in']



class DispatchWrap:
    def __init__(self, plant, params):

        self.plant = plant             # Plant design and operating properties
        self.params = params           # Parameters, needs to include everything in dispatch_wrap_params and mediator_params

        self.default_disp_stored_vars = [
            'cycle_on', 'cycle_standby', 'cycle_startup', 'receiver_on', 'receiver_startup', 'receiver_standby', 
            'receiver_power', 'thermal_input_to_cycle', 'electrical_output_from_cycle', 'net_electrical_output',
            'tes_soc', 'yrsd', 'ursd', 'receiver_outlet_temp', 'hot_salt_tank_temp', 'cold_salt_tank_temp']

        self.user_defined_cycle_input_file = '../../librtdispatch/udpc_noTamb_dependency.csv'  # Only required if cycle_type is user_defined

        self.price_data = params['dispatch_factors_ts']   # Electricity prices ($/MWh) at ssc time resolution

        ## DISPATCH PERSISTING INTERMEDIARIES ############################################################################################################
        self.dispatch_params = DispatchParams()         # Structure to contain all inputs for dispatch model 
        self.first_run = True                           # Is this the first time run() is called?
        self.ursd0 = 0
        self.yrsd0 = 0
        self.current_day_schedule = []                  # Committed day-ahead generation schedule for current day (MWe)
        self.next_day_schedule = []                     # Predicted day-ahead generation schedule for next day (MWe)
        self.date_for_current_day_schedule = None       # Datetime of first point in schedule stored in self.current_day_schedule
        self.date_for_next_day_schedule = None          # Datetime of first point in schedule stored in self.next_day_schedule
        
        ## SSC OUTPUTS ###################################################################################################################################
        self.results = None                             # Results

        #TODO: These aren't currently used.  Do we need to calculate and return these from within the dispatch model?
        '''
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
        '''

        return

    #--- Run simulation
    def run(self, datetime_start, ssc_horizon, weather_dataframe, 
            f_estimates_for_dispatch_model, update_interval, initial_plant_state):
        '''
        Assumes that datetime_start and weather_dataframe are not local time, and correspond to a constant offset used in ssc (same as used in tech_wrap.simulate)
        ssc_horizon = time horizon of dispatch targets that will be utilized in ssc (used only to condense dispatch target arrays to the correct length)
        update_interval = frequency (hr) at which dispatch optimization will be re-run
        '''
        # Notes: The ssc simulation time resolution is assumed to be <= the shortest dispatch time step
        #         All dispatch time steps and time horizons are assumed to be an integer multiple of the ssc time step

        weather_timestep = (weather_dataframe.index[1]-weather_dataframe.index[0]).total_seconds()    # Time step in weather data frame (s)
       
        #--- Define horizon for this call to the dispatch optimization horizon (use specified 'dispatch_horizon' unless weather data is not available for the full horizon)
        weather_horizon = (weather_dataframe.index[-1]-weather_dataframe.index[0]).total_seconds() + weather_timestep  # Total duration of weather available for simulation relative to simulation start_date (s)
        horizon = int(min(weather_horizon, self.params['dispatch_horizon']*3600)/3600.)    # hr        

        npts_horizon = int(horizon*self.params['time_steps_per_hour'])
        clearsky = np.nan_to_num([weather_dataframe['Clear Sky DNI'][i] for i in range(npts_horizon)], nan = 0.0)
        
        #--- Define time step arrays for dispatch optimization
        dispatch_steplength_end_time = self.params['dispatch_steplength_end_time']
        if horizon != self.params['dispatch_horizon']:
            horizon_minutes = int(horizon*60 / self.params['dispatch_steplength_array'][0]) * self.params['dispatch_steplength_array'][0]  # Horizon rounded to nearest integer multiple of shortest dispatch time step
            end_time_minutes = calculate_time_steps(horizon_minutes, self.params['dispatch_steplength_array'], [t*60 for t in self.params['dispatch_steplength_end_time']])
            dispatch_steplength_end_time  = [t/60 for t in end_time_minutes]

        #--- Initialize schedules, plant state, shut-down state parameters, and clear-sky data
        retvars = self.default_disp_stored_vars
        if self.first_run == True:
            self.current_day_schedule = []
            self.next_day_schedule = []
            self.date_for_current_day_schedule = None
            self.date_for_next_day_schedule = None
            ursd0 = None
            yrsd0 = None
            self.first_run == False
        else:
            ursd0 = self.ursd0  # Receiver accumulated energy for shutdown (taken from last dispatch solution for now)
            yrsd0 = self.yrsd0  # Receiver binary shutdown state (taken from last dispatch solution for now)       
                 
        self.plant.state = initial_plant_state

        #--- Calculate time-related values
        tod = int(get_time_of_day(datetime_start))        # Current time of day (s)
        toy = int(get_time_of_year(datetime_start))       # Current time of year (s)    
        nph = int(self.params['time_steps_per_hour'])     # Number of time steps per hour
        startpt = int(toy/3600)*nph                       # Start point in annual arrays
        sscstep = 3600/nph                                # ssc time step (s)
        date_today = datetime.datetime(datetime_start.year, datetime_start.month, datetime_start.day)

        #--- Update stored day-ahead generation schedule for current day (if relevant)
        if self.params['use_day_ahead_schedule']:

            if self.params['day_ahead_schedule_from'] == 'external' and len(self.current_day_schedule)==0:
                self.current_day_schedule = self.get_external_day_ahead_schedule(datetime_start)   # TODO: Not currently set up.  Do we need to keep this functionality?

            elif self.params['day_ahead_schedule_from'] == 'calculated' and self.date_for_current_day_schedule != date_today:   # The schedule stored in current_day_schedule is not for today (or one doesn't exist)
                if self.date_for_next_day_schedule == date_today:    # Schedule stored in next_day_schedule is for today
                    self.current_day_schedule = deepcopy(self.next_day_schedule)
                    self.date_for_current_day_schedule = self.date_for_next_day_schedule
                    self.next_day_schedule = []
                    self.date_for_next_day_schedule = None
                else:
                    self.current_day_schedule = []
                    self.date_for_current_day_schedule = None

        #--- Run ssc for dispatch estimates
        npts_horizon = int(horizon* nph)
        ssc_estimates = f_estimates_for_dispatch_model(
            plant_state = self.plant.state,
            datetime_start = datetime_start,
            horizon = horizon,     
            weather_dataframe = weather_dataframe,
        )

        if 'clearsky' not in ssc_estimates or max(ssc_estimates['clearsky']) < 1.e-3:   # Clear-sky data wasn't passed through ssc (likely because ssc controlled from actual DNI)
            ssc_estimates['clearsky'] = clearsky


        #--- Create dispatch optimization model inputs for this time horizon using ssc estimates
        include_day_ahead_in_dispatch = len(self.current_day_schedule)>0  # Only use day ahead schedule in dispatch model if one exists
        dispatch_model_inputs = self.setup_dispatch_model(
            datetime_start = datetime_start, 
            ssc_estimates = ssc_estimates,
            horizon = horizon,      
            dispatch_steplength_array = self.params['dispatch_steplength_array'],
            dispatch_steplength_end_time = dispatch_steplength_end_time,
            clearsky_data = clearsky,
            price = self.price_data[startpt:startpt+npts_horizon],          # [$/MWh] Update prices for this horizon
            ursd0 = ursd0,
            yrsd0 = yrsd0
        )

        #--- Run dispatch optimization 
        include = {"pv": False, "battery": False, "persistence": False, "force_cycle": False, "op_assumptions": False,
                    "signal":include_day_ahead_in_dispatch, "simple_receiver": False}
            
        dispatch_soln = run_dispatch_model(dispatch_model_inputs, include)

        #--- Populate results
        if dispatch_soln:    # Dispatch model was successful
            Rdisp_all = dispatch_soln.get_solution_at_ssc_steps(self.dispatch_params, sscstep/3600., horizon)
            Rdisp = {'disp_'+key:value for key,value in Rdisp_all.items() if key in retvars}
            Rdisp['disp_Qu'] = self.dispatch_params.Qu
            Rdisp['disp_Eu'] = self.dispatch_params.Eu
            Rdisp['disp_Wdotu'] = self.dispatch_params.Wdotu

            # Update calculated schedule for next day (if relevant). Assume schedule must be committed within 1hr of designated time (arbitrary...)
            if self.params['use_day_ahead_schedule'] and self.params['day_ahead_schedule_from'] == 'calculated':
                if tod/3600 >= self.params['day_ahead_schedule_time'] and tod/3600 < self.params['day_ahead_schedule_time']+1:            # Within the time window to commit next-day schedule
                    if len(self.next_day_schedule)==0 or self.date_for_next_day_schedule != date_today + datetime.timedelta(hours = 24):  # Next-day schedule doesn't already exist (or one exists, but is left-over from a different day)
                        self.next_day_schedule = get_day_ahead_schedule(
                            day_ahead_schedule_steps_per_hour = self.params['day_ahead_schedule_steps_per_hour'],
                            Delta = self.dispatch_params.Delta,
                            Delta_e = self.dispatch_params.Delta_e,
                            net_electrical_output = dispatch_soln.net_electrical_output,
                            day_ahead_schedule_time = self.params['day_ahead_schedule_time']
                            )
                        if len(self.next_day_schedule)>0:
                            self.date_for_next_day_schedule = date_today + datetime.timedelta(hours = 24)

            #--- Set ssc dispatch targets
            ssc_dispatch_targets = DispatchTargets(dispatch_soln, self.plant, self.dispatch_params, sscstep, ssc_horizon.total_seconds()/3600.)

            #--- Get shut-down state parameters at the point in time in this dispatch solution when the next dispatch call will occur (stand-in for plant state as ssc does not model reciever shutdown)
            # TODO: should eventually remove this is favor of data from plant database (if possible)... shut down state from dispatch model won't match real life unless real plant follows this schedule
            # Note these are only important if shutdown requirements (plant.design['rec_sd_delay'] and plant.design['q_rec_shutdown_fraction']) are nonzero
            self.ursd0 = dispatch_soln.get_value_at_time(self.dispatch_params, update_interval, 'ursd')      # set to False when it doesn't exist 
            self.yrsd0 = dispatch_soln.get_value_at_time(self.dispatch_params, update_interval, 'yrsd')      # set to False when it doesn't exist


        else:  # Infeasible solution was returned, revert back to running ssc without dispatch targets
            Rdisp = None
            ssc_dispatch_targets = None

        dispatch_outputs = {
            'ssc_dispatch_targets': ssc_dispatch_targets,
            'Rdisp': Rdisp,
	        'current_day_schedule': list(self.current_day_schedule),
	        'next_day_schedule': list(self.next_day_schedule)
        }

        self.results = dispatch_outputs

        return dispatch_outputs


    def get_nonlinear_horizon(self, clearsky_data):
        nonlinear_time = self.params['nonlinear_model_time']   # Nominal time horizon for nonlinear model (hr)
        if self.params['use_linear_dispatch_at_night']:
            endpt = int(nonlinear_time * self.params['time_steps_per_hour'])  # Last point in clear-sky array corresponding to nonlinear portion of dispatch model
            if clearsky_data[0:endpt].max() <= self.params['night_clearsky_cutoff']:
                nonlinear_time = 0.0
        return nonlinear_time

    def setup_dispatch_model(self, datetime_start, ssc_estimates, horizon, dispatch_steplength_array, dispatch_steplength_end_time,
        clearsky_data, price, ursd0, yrsd0):
        '''horizon in [hr]'''
        #--- Set dispatch optimization properties for this time horizon using ssc estimates

        tod = int(get_time_of_day(datetime_start))                     # Current time of day (s) 
        sscstep = 3600/int(self.params['time_steps_per_hour'])         # ssc time step (s)

        #--- Set time steps for dispatch model
        nonlinear_time = self.get_nonlinear_horizon(clearsky_data)  # Get time horizon for nonlinear model (this is not used as of 6/2021, only the linear model is currently available)
        self.dispatch_params.set_dispatch_time_arrays(dispatch_steplength_array, dispatch_steplength_end_time,
            horizon, nonlinear_time, self.params['disp_time_weighting'])    

        #--- Set plant design and plant state in dispatch parameters
        self.dispatch_params.set_fixed_parameters_from_plant_design(self.plant)  # Only uses fixed plant design parameters and not plant state parameters, could be moved
        self.dispatch_params.set_initial_state(self.plant)                       # TODO: Make sure plant includes plant_state at this point  plant or plant.state?
        self.dispatch_params.disp_time_weighting = self.params['disp_time_weighting']
        
        #--- Update approximate receiver shutdown state from previous dispatch solution (not returned from ssc)
        self.dispatch_params.set_approximate_shutdown_state_parameters(self.plant.state, ursd = ursd0, yrsd = yrsd0)  # Set initial state parameters related to shutdown from dispatch model (because this cannot be derived from ssc)
    
        #--- Set time-indexed parameters in dispatch inputs
        self.dispatch_params.set_default_grid_limits()
        self.dispatch_params.P = translate_to_variable_timestep([p/1000. for p in price], sscstep/3600., self.dispatch_params.Delta)  # $/kWh
        self.dispatch_params.avg_price = self.params['avg_price']/1000.
        self.dispatch_params.avg_price_disp_storage_incentive = self.params['avg_price_disp_storage_incentive'] / 1000.  # $/kWh  # Only used in storage inventory incentive -> high values cause the solutions to max out storage rather than generate electricity
        self.dispatch_params.avg_purchase_price = self.params['avg_purchase_price']/1000    # $/kWh 
        self.dispatch_params.day_ahead_tol_plus = self.params['day_ahead_tol_plus']*1000    # kWhe
        self.dispatch_params.day_ahead_tol_minus = self.params['day_ahead_tol_minus']*1000  # kWhe

        self.dispatch_params.set_estimates_from_ssc_data(self.plant, ssc_estimates, sscstep/3600.) 
        
        #--- Set day-ahead schedule in dispatch parameters
        if len(self.current_day_schedule)>0:
            day_ahead_horizon = 24 - int(tod/3600)   # Number of hours of day-ahead schedule to use.  This probably only works in the dispatch model if time horizons are updated at integer multiples of an hour
            use_schedule = [self.current_day_schedule[s]*1000 for s in range(24-day_ahead_horizon, 24)]   # kWhe
            self.dispatch_params.set_day_ahead_schedule(use_schedule, self.params['day_ahead_pen_plus']/1000, self.params['day_ahead_pen_minus']/1000)
            
        #--- Create copy of params, and convert all lists and numpy arrays into dicts, where each value has a key equal to the index + 1 (required format for pyomo dispatch model)
        dispatch_model_inputs = self.dispatch_params.copy_and_format_indexed_inputs()     
        dispatch_model_inputs.transition = 0   #Transition index between nonlinear/linear models. TODO: Once nonlinear model is available, need to update this to correspond to nonlinear_time 

        return dispatch_model_inputs



def run_dispatch_model(dispatch_model_inputs, include):
    rt = dispatch_model.RealTimeDispatchModel(dispatch_model_inputs, include)
    rt_results = rt.solveModel()
    
    if rt_results.solver.termination_condition == TerminationCondition.infeasible:
        return False

    if True: #test two-phased approach as default for now.  TODO: add new param in dispatch_model_inputs asking about 2-phase
        dispatch_model_inputs.transition = 4
        rt2 = dispatch_model.RealTimeDispatchModel(dispatch_model_inputs, include)
        rt2.populate_variable_values(rt)
        rt2.fix_binaries()
        rt2.solveModel(solver='ipopt')
        if rt_results.solver.termination_condition == TerminationCondition.infeasible:
            return False
        return DispatchSoln(rt2.model)
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
            next_day_schedule = translate_to_fixed_timestep(wnet[inds], disp_steps[inds], day_ahead_step) 
        return next_day_schedule
    else:
        return []


def get_weather_at_day_ahead_schedule(weather_df, npts_horizon):
    return {k: list(weather_df[k][0:npts_horizon])  for k in ['DNI', 'Temperature', 'Wind Speed']}


def calculate_time_steps(horizon, step_size, target_step_end_time):
    # Set number of each dispatch time step to maintain the largest number of each step (within the target number of steps), subject to integer requirements on the number of steps of each duration
    # Assumes: (1) All time step lengths are an integer multiple of the shortest time step
    #          (2) horizon is an integer multiple of the smallest time step size
    #          (2) Step sizes increase monotonically
    
    target_steps, steps, end_time = [np.zeros_like(step_size) for j in range(3)]
    for j in range(len(step_size)):
        start = 0 if j == 0 else target_step_end_time[j-1]
        target_steps[j] = int((target_step_end_time[j] - start) / step_size[j])   # Target number of steps of each length
        
    remaining_horizon = horizon
    for j in range(len(step_size)-1):
        n2 = (remaining_horizon - step_size[j]*target_steps[j]) / step_size[j+1]  # Number of steps (s+1) within remaining horizon after target number of steps (s)
        n = int((remaining_horizon - ceil(n2)*step_size[j+1]) / step_size[j])     # Number of steps (s) that are possible to maintain integer number of remaining steps
        steps[j] = n
        start = 0 if j == 0 else end_time[j-1]
        end_time[j] = start + n*step_size[j]
        remaining_horizon -= n * step_size[j]     
    steps[-1] = int(remaining_horizon / step_size[-1])
    end_time[-1] = end_time[-2] + steps[-1]* step_size[-1]
    if horizon != end_time[-1]:
        print ('WARNING: Dispatch model time step arrays do not match required horizon')
    return end_time.tolist()




def get_time_of_day(date):
    return (date- datetime.datetime(date.year,date.month,date.day,0,0,0,tzinfo=date.tzinfo)).total_seconds()
        
def get_time_of_year(date):
    return (date - datetime.datetime(date.year,1,1,0,0,0,tzinfo=date.tzinfo)).total_seconds()

# Update annual array to a new timestep (assuming integer multiple of new timesteps in old timestep or vice versa)
def translate_to_new_timestep(data, old_timestep, new_timestep):
    n = len(data)
    if new_timestep > old_timestep:  # Average over consecutive timesteps
        nperavg = int(new_timestep / old_timestep)
        nnew = int(n/nperavg)
        newdata = np.reshape(np.array(data), (nnew, nperavg)).mean(1)
    else:  # Repeat consecutive timesteps
        nrepeat = int(old_timestep / new_timestep)
        newdata = np.repeat(data, nrepeat)
    return newdata.tolist()

# Translate arrays from a fixed timestep (dt_fixed) to variable timestep (dt_var)
# Assumes that all variable timesteps are an integer multiple of the fixed timstep, or vice versa, and that end points of fixed and variable timesteps coincide
def translate_to_variable_timestep(data, dt_fixed, dt_var):
    n = len(dt_var)  
    dt_fixed_sec = int(ceil(dt_fixed*3600 - 0.0001))
    data_var = np.zeros(n)
    s = 0
    j = 0
    while j<n:
        dt_sec = int(ceil(dt_var[j]*3600 - 0.0001))
        if dt_sec > dt_fixed_sec:  # Variable timestep is larger than fixed timestep, apply average of all fixed timesteps contained within the variable timestep
            n_fixed_per_var = int(dt_sec / dt_fixed_sec)
            for i in range(n_fixed_per_var):
                data_var[j] += data[s+i]/n_fixed_per_var 
            j +=1
            s += n_fixed_per_var
        else:  # Variable timestep is shorter than fixed timestep, repeat fixed timestep value for all variable timesteps within the interval
            t = 0
            while t < dt_fixed_sec - 0.0001:
                data_var[j] = data[s]
                t += dt_var[j]*3600
                j+=1
            s+=1
    return data_var


# Translate arrays from a variable timestep (dt_var) to a fixed timestep (dt_fixed)
# Assumes that all variable timesteps are an integer multiple of the fixed timstep, or vice versa, and that end points of fixed and variable timesteps coincide
def translate_to_fixed_timestep(data, dt_var, dt_fixed):
    n = len(dt_var)
    dte = np.cumsum(dt_var)  
    dt_fixed_sec = int(ceil(dt_fixed*3600 - 0.0001))
    horizon = int(ceil(dte[-1]* 3600 - 0.0001))  # Full dispatch time horizon (s)
    n_fixed = int(horizon / dt_fixed_sec)  # Number of fixed time steps in horizon
    data_fixed = np.zeros(n_fixed)
    s = 0
    j = 0
    while j<n:
        dt_sec = int(ceil(dt_var[j]*3600 - 0.0001))
        if dt_sec >= dt_fixed_sec:  # Variable timestep is larger than fixed timestep, repeat value at variable timestep for each fixed timestep in the interval
            n_per_var = int(dt_sec / dt_fixed_sec)  
            for i in range(n_per_var):
                data_fixed[s+i] = data[j]
            s += n_per_var 
            j+=1

        else:  # Fixed timestep is larger than variable timestep, apply average of all variable timesteps contained within fixed timestep
            t = 0
            while t < dt_fixed_sec - 0.0001:
                data_fixed[s] += data[j] * (dt_var[j] / dt_fixed)
                t += dt_var[j]*3600
                j+=1
            s+=1
    return data_fixed

def interpret_user_defined_cycle_data(ud_ind_od):
    data = np.array(ud_ind_od)
        
    i0 = 0
    nT = np.where(np.diff(data[i0::,0])<0)[0][0] + 1 
    Tpts = data[i0:i0+nT,0]
    mlevels = [data[j,1] for j in [i0,i0+nT,i0+2*nT]]
    
    i0 = 3*nT
    nm = np.where(np.diff(data[i0::,1])<0)[0][0] + 1 
    mpts = data[i0:i0+nm,1]
    Tamblevels = [data[j,2] for j in [i0,i0+nm,i0+2*nm]]
    
    i0 = 3*nT + 3*nm
    nTamb = np.where(np.diff(data[i0::,2])<0)[0][0] + 1 
    Tambpts = data[i0:i0+nTamb,2]
    Tlevels = [data[j,0] for j in [i0,i0+nm,i0+2*nm]]
    
    return {'nT':nT, 'Tpts':Tpts, 'Tlevels':Tlevels, 'nm':nm, 'mpts':mpts, 'mlevels':mlevels, 'nTamb':nTamb, 'Tambpts':Tambpts, 'Tamblevels':Tamblevels}



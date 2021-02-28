
from math import ceil, pi, log, isnan
import numpy as np
import util
from copy import deepcopy
import ssc_wrapper
import pyomo.environ as pe
import dispatch_model
from pyomo.opt import TerminationCondition

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
    def set_fixed_parameters_from_plant_design(self, design, properties):
        q_pb_design = design.get_cycle_thermal_rating()  #MWt
        m_pb_design = design.get_cycle_design_mass_flow()   # kg/s
        q_rec_design = design.Qrec  # MWt
        m_rec_design = design.get_receiver_design_mass_flow() #kg/s
        nhel = design.get_number_of_heliostats()
        
        m_active_hot_max, m_active_cold_max, m_inactive_hot, m_inactive_cold = design.get_storage_mass()
        
        # Receiver parameters
        self.Drsu = properties.rec_su_delay 
        self.Drsd = properties.rec_sd_delay  
        self.Er = properties.rec_qf_delay * q_rec_design * 1000. 
        self.Qrl = properties.f_rec_min * q_rec_design * 1000.
        self.Qrsb = properties.q_rec_standby_fraction * q_rec_design * 1000.
        self.Qrsd = properties.q_rec_shutdown_fraction * q_rec_design * 1000   
        self.Qru = self.Er / properties.rec_su_delay
        self.mdot_r_min = properties.f_rec_min * m_rec_design
        self.mdot_r_max = properties.csp_pt_rec_max_oper_frac * m_rec_design  
        self.T_rout_min = properties.T_rout_min  
        self.T_rout_max = properties.T_rout_max 

        # TES parameters
        self.Eu = q_pb_design * design.tshours  * 1000.
        self.Cp = design.get_cp_htf(0.5*(design.T_htf_cold_des+design.T_htf_hot_des)) * 1.e-3  
        self.mass_cs_min = properties.mass_cs_min_frac * m_active_cold_max
        self.mass_cs_max = m_active_cold_max
        self.mass_hs_min = properties.mass_hs_min_frac * m_active_hot_max
        self.mass_hs_max = m_active_hot_max
        self.T_cs_min = properties.T_cs_min 
        self.T_cs_max = properties.T_cs_max
        self.T_hs_min = properties.T_hs_min 
        self.T_hs_max = properties.T_hs_max   
        self.T_cs_des = design.T_htf_cold_des
        self.T_hs_des = design.T_htf_hot_des
        
        # Cycle parameters
        self.Ec = properties.startup_frac * q_pb_design * 1000.  
        self.Ew = properties.startup_frac_warm * q_pb_design * 1000. 
        self.eta_des = design.design_eff    # TODO: Probably good enough for now. ssc calls the power cycle model at full load and design point ambient T to exactly match full-load performance
        self.Qb = properties.q_sby_frac * q_pb_design * 1000.     
        self.Qc = self.Ec / ceil(properties.startup_time / min(self.Delta)) / min(self.Delta)      # TODO: Not clear how to best define this with variable time steps.  Using minimum time step for maximum allowable startup energy rate
        self.Ql = properties.cycle_cutoff_frac * q_pb_design * 1000. 
        self.Qu = properties.cycle_max_frac * q_pb_design * 1000.   
        self.kl = m_pb_design * properties.cycle_cutoff_frac * self.Cp 
        self.ku = m_pb_design * properties.cycle_max_frac * self.Cp 
        self.Wdot_design = q_pb_design * design.design_eff * 1000.  
        self.Wdot_p_max = 25000.  # TODO: Fixing this for now, but probably should base off of design parameters
        self.mdot_c_design = m_pb_design
        self.mdot_c_min = properties.cycle_cutoff_frac*m_pb_design 
        self.mdot_c_max = properties.cycle_max_frac*m_pb_design  
        self.T_cin_design = design.T_htf_hot_des 
        self.T_cout_min = properties.T_cs_min 
        self.T_cout_max = properties.T_cs_max
        self.delta_T_design = design.T_htf_hot_des - design.T_htf_cold_des
        self.delta_T_max = max(abs(properties.alpha_b * self.delta_T_design), properties.T_hs_max - properties.T_cs_min)

        if design.pc_config == 1: # User-defined cycle
            self.set_linearized_params_from_udpc_inputs(design, properties)
        else:
            print ('Warning: Dispatch optimization parameters are currently only set up for user-defined power cycle. Defaulting to constant efficiency vs load')
            self.etap = self.eta_des  
            self.Wdotl = self.Ql*self.eta_des  
            self.Wdotu = self.Qu*self.eta_des
    
        self.W_delta_plus = (properties.pc_rampup) * self.Wdotu 
        self.W_delta_minus = (properties.pc_rampdown) * self.Wdotu 
        self.W_v_plus = (properties.pc_rampup_vl) * self.Wdotu 
        self.W_v_minus = (properties.pc_rampdown_vl) * self.Wdotu 
        self.Yu = properties.Yu  
        self.Yd = properties.Yd  
        
        # Parastic loads
        self.Ehs = properties.p_start * nhel 
        self.Wh_track = properties.p_track * nhel    
        self.Wh_comm = properties.p_comm * nhel  
        self.estimate_receiver_pumping_parasitic(design)  # Sets Lr, Pr
        #self.Lr = properties.Lr      
        #self.Pr = properties.Pr       
        self.Wht_full = properties.Wht_fract * design.Qrec * 1000.
        self.Wht_part = properties.Wht_fract_partload * design.Qrec * 1000.
        self.Lc = properties.pb_pump_coef * m_pb_design / (q_pb_design * 1000) 
        self.Wb = properties.Wb_fract* design.P_ref*1000.
        self.Wc = properties.Wc_fract* design.P_ref*1000.
        
        # Cost parameters
        self.alpha = 1.0  
        self.Crec = properties.Crec   
        self.Crsu = properties.Crsu  
        self.Crhsp = properties.Crhsp     
        self.Cpc = properties.Cpc
        self.Ccsu = properties.Ccsu 
        self.Cchsp = properties.Cchsp
        self.C_delta_w = properties.C_delta_w
        self.C_v_w  = properties.C_v_w  
        self.Ccsb = properties.Ccsb 
        
        # Indexing and piecewise-linear indexed parameters
        self.nc = len(properties.Pc)
        self.Pc = properties.Pc 
        self.Bc = properties.Bc      
        self.nfw = len(properties.Pfw)
        self.Pfw = properties.Pfw    
        self.Bfw = properties.Bfw 
        
        # Other parameters
        self.alpha_b = properties.alpha_b
        self.alpha_T = properties.alpha_T
        self.alpha_m = properties.alpha_m
        self.beta_b = properties.beta_b
        self.beta_T = properties.beta_T
        self.beta_m = properties.beta_m
        self.beta_mT = properties.beta_mT

        return
        
  
    def estimate_receiver_pumping_parasitic(self, design, nonheated_length = 0.2):
        m_rec_design = design.get_receiver_design_mass_flow() #kg/s
        Tavg = 0.5*(design.T_htf_cold_des + design.T_htf_hot_des)
        rho = design.get_density_htf(Tavg)
        visc = design.get_visc_htf(Tavg)

        npath = 1
        nperpath = design.N_panels
        if design.Flow_type == 1 or design.Flow_type == 2:
            npath = 2
            nperpath = int(design.N_panels/2)
        elif design.Flow_type == 9:
            npath = int(design.N_panels/2)
            nperpath = 2
            
        ntube = int(pi * design.D_rec/design.N_panels / (design.d_tube_out*1.e-3))  # Number of tubes per panel
        m_per_tube = m_rec_design / npath / ntube  # kg/s per tube
        tube_id = (design.d_tube_out - 2*design.th_tube) / 1000.  # Tube ID in m
        Ac = 0.25*pi*(tube_id**2)
        vel = m_per_tube / rho / Ac  # HTF velocity
        Re = rho * vel * tube_id / visc
        eD = 4.6e-5 / tube_id
        ff = (-1.737*log(0.269*eD - 2.185/Re*log(0.269*eD+14.5/Re)))**-2
        fd = 4*ff 
        Htot = design.rec_height* (1+nonheated_length)
        dp = 0.5*fd*rho*(vel**2) * (Htot/tube_id + 4*30 + 2*16) * nperpath  # Frictional pressure drop (Pa) (straight tube, 90deg bends, 45def bends)
        dp += rho * 9.8 * design.h_tower  # Add pressure drop from pumping up the tower
        if nperpath%2 == 1:   
            dp += rho * 9.8 * Htot  
            
        wdot = dp * m_rec_design / rho / design.eta_pump / 1.e6   # Pumping parasitic at design point reciever mass flow rate (MWe)
        
        self.Lr = wdot / design.Qrec # MWe / MWt
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
    
    def set_initial_state(self, design, state):
        m_des = design.get_design_storage_mass()

        m_hot = (state.csp_pt_tes_init_hot_htf_percent/100) * m_des  # Available active mass in hot tank
        m_cold = ((100 - state.csp_pt_tes_init_hot_htf_percent)/100) * m_des   # Available active mass in cold tank
        cp = design.get_cp_htf(0.5*(state.T_tank_hot_init+design.T_htf_cold_des)) # J/kg/K

        self.T_cs0 = min(max(self.T_cs_min, state.T_tank_cold_init), self.T_cs_max)
        self.T_hs0 = min(max(self.T_hs_min, state.T_tank_hot_init), self.T_hs_max)    
        self.s0 = min(self.Eu,  m_hot * cp * (state.T_tank_hot_init - design.T_htf_cold_des) * 1.e-3 / 3600)  # Note s0 is calculated internally in the pyomo dispatch model
        self.mass_cs0 = min(max(self.mass_cs_min, m_cold), self.mass_cs_max)
        self.mass_hs0 = min(max(self.mass_hs_min, m_hot), self.mass_hs_max)
        max_allowable_mass = 0.995*self.Eu*3600 /self.Cp/(self.T_hs0 - self.T_cs_des) + self.mass_hs_min  # Max allowable mass for s0 = Eu from dispatch model s0 calculation in pyomo
        self.mass_hs0 = min(self.mass_hs0, max_allowable_mass)
        self.wdot0 = state.wdot0 * 1000.          
        
        
        self.yr0 = (state.rec_op_mode_initial == 2)
        self.yrsb0 = False      # TODO: no official receiver "standby" mode currently exists in ssc.  Might be able to use the new cold-tank recirculation to define this
        self.yrsu0 = (state.rec_op_mode_initial == 1)
        self.yrsd0 = False     # TODO: no official receiver "shutdown" mode currently exists in ssc. 
        self.y0 = (state.pc_op_mode_initial == 1) 
        self.ycsb0 = (state.pc_op_mode_initial == 2) 
        self.ycsu0 = (state.pc_op_mode_initial == 0 or state.pc_op_mode_initial == 4) 

        self.drsu0 = state.disp_rec_persist0 if self.yrsu0 else 0.0   
        self.drsd0 = state.disp_rec_persist0 if state.rec_op_mode_initial == 0 else 0.0 # TODO: defining time in shutdown mode as time "off", will this work in the dispatch model?
        self.Yu0 = state.disp_pc_persist0 if self.y0 else 0.0
        self.Yd0 = state.disp_pc_off0 if (not self.y0) else 0.0
        
        
        # Initial startup energy accumulated
        if isnan(state.pc_startup_energy_remain_initial):  # ssc seems to report nan when startup is completed
            self.ucsu = self.Ec
        else:   
            self.ucsu0 = max(0.0, self.Ec - state.pc_startup_energy_remain_initial) 
            if self.ucsu0 > (1.0 - 1.e-6)*self.Ec:
                self.ucsu0 = self.Ec
            
        rec_accum_time = max(0.0, self.Drsu - state.rec_startup_time_remain_init)
        rec_accum_energy = max(0.0, self.Er - state.rec_startup_energy_remain_init/1000.)
        self.ursu0 = min(rec_accum_energy, rec_accum_time * self.Qru)  # Note, SS receiver model in ssc assumes full available power is used for startup (even if, time requirement is binding)
        if self.ursu0 > (1.0 - 1.e-6)*self.Er:
            self.ursu0 = self.Er

        self.ursd0 = 0.0   #TODO: How can we track accumulated shut-down energy (not modeled in ssc)

        return
    
    # Approximate initial state of receiver "shutdown" variables using yrsd and ursd at last time point accepted from previous dispatch solution
    def set_approximate_shutdown_state_parameters(self, state, ursd = 0.0, yrsd = 0):
        if state.rec_op_mode_initial == 3:  # Receiver is off, use ursd, yrsd from previous dispatch solution to define initial properties
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
        if design.pc_config == 1: # User-defined cycle
            # TODO: Note that current user-defined cycle neglects ambient T effects
            Tdry = util.translate_to_variable_timestep(S['tdry'], sscstep, self.Delta)
            etamult, wmult = self.get_ambient_T_corrections_from_udpc_inputs(design, Tdry)
     
            #TODO: Make sure these should be efficiency values and not mulipliers
            self.etaamb  = etamult * design.design_eff
            self.etac = wmult * design.ud_f_W_dot_cool_des/100.

            
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
    

    def set_linearized_params_from_udpc_inputs(self, design, properties):
        q_pb_design = design.get_cycle_thermal_rating()  #MWt
        D = util.interpret_user_defined_cycle_data(design.ud_ind_od)
        eta_adj_pts = [design.ud_ind_od[p][3]/design.ud_ind_od[p][4] for p in range(len(design.ud_ind_od)) ]
        xpts = D['mpts']
        step = xpts[1] - xpts[0]
        
        # Interpolate for cycle performance at specified min/max load points
        fpts = [properties.cycle_cutoff_frac, properties.cycle_max_frac]
        q, eta = [ [] for v in range(2)]
        for j in range(2):
            p = max(0, min(int((fpts[j] - xpts[0]) / step), len(xpts)-2) )  # Find first point in user-defined array of load fractions for interpolation
            i = 3*D['nT'] + D['nm'] + p    # Index of point in full list of udpc points (at design point ambient T)
            eta_adj = eta_adj_pts[i] + (eta_adj_pts[i+1] - eta_adj_pts[i])/step * (fpts[j] - xpts[p])
            eta.append(eta_adj * design.design_eff)
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
        D = util.interpret_user_defined_cycle_data(design.ud_ind_od)
        
        Tambpts = np.array(D['Tambpts'])
        i0 = 3*D['nT']+3*D['nm']+D['nTamb']  # first index in udpc data corresponding to performance at design point HTF T, and design point mass flow
        npts = D['nTamb']
        etapts = [ design.ud_ind_od[j][3]/design.ud_ind_od[j][4] for j in range(i0, i0+npts)]
        wpts = [ design.ud_ind_od[j][5] for j in range(i0, i0+npts)]
        
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





#=============================================================================
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


def setup_dispatch_model(R_est, freq, horizon, include_day_ahead_in_dispatch,
    dispatch_params, plant_design, plant_state, nonlinear_model_time, use_linear_dispatch_at_night,
    clearsky_data, night_clearky_cutoff, dispatch_steplength_array, dispatch_steplength_end_time,
    disp_time_weighting, price, sscstep, avg_price, avg_price_disp_storage_incentive,
    avg_purchase_price, day_ahead_tol_plus, day_ahead_tol_minus,
    tod, current_day_schedule, day_ahead_pen_plus, day_ahead_pen_minus,
    dispatch_horizon, night_clearsky_cutoff, properties, ursd_last, yrsd_last):

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
    dispatch_params.set_approximate_shutdown_state_parameters(plant_state, ursd = ursd_last, yrsd = yrsd_last)  # Set initial state parameters related to shutdown from dispatch model (because this cannot be derived from ssc)

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


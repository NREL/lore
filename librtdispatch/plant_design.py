
import numpy as np
from math import pi


class PlantDesign:
    def __init__(self):

        # System
        self.T_htf_cold_des = 290.0 # C
        self.T_htf_hot_des = 574.0 # C
        self.P_ref = 115  # MW
        self.gross_net_conversion_factor = 0.90
        self.design_eff = 0.412
        self.Qrec = 669.9 # MWt
        self.tshours = 10.0 #10.0 # hr (10.1 results in a close match of total salt mass compared to CD data)(TODO: using 10 hrs, SSC hangs during simulation??)
        self.solarm = self.Qrec * self.design_eff / self.P_ref
        
        # Solar field
        self.heliostat_field_file = './input_files/CD_processed/Crescent_Dunes_heliostat_layout.csv' # TODO: This the field layout with a fixed z-coordinate.  Could update to call through SolarPILOT API with actual z-coordinates? (but model validation cases suggest this is not terribly important)
        self.helio_positions = []
        self.N_hel = 0
        self.helio_height = 12.2 # m
        self.helio_width = 12.2 # m
        self.n_facet_x = 2
        self.n_facet_y = 8
        self.dens_mirror = 0.97
        
        # Receiver
        self.rec_htf = 17  # 17 = 60% NaNO3, 40% KNO3 nitrate salt
        self.rec_height = 21.6029    # Heated panel length (m)
        self.D_rec = 17.65  # m
        self.d_tube_out = 40 # mm
        self.th_tube = 1.25 # mm
        self.N_panels = 20
        self.h_tower = 193.458 # m
        self.mat_tube = 2   # 2 = SS316, 32 = H230, 33 = 740H
        self.piping_length_mult = 2.6   # TODO: Better value for CD?  Piping length multiplier
        self.Flow_type = 1
        self.crossover_shift = 1
        self.eta_pump = 0.85    # Receiver pump efficiency

        # TES
        self.h_tank = 12.0 # m
        self.h_tank_min = 1.0 # m  # Minimum allowable HTF height in storage tank (m)
        self.hot_tank_Thtr = 500 # C
        self.cold_tank_Thtr = 280 # C
        
        
        # Cycle
        self.pc_config = 0 # 0=Steam Rankine, 1=User-defined
        self.ud_f_W_dot_cool_des = 0.0  # This is an estimate for the cooling system (700 kW) TODO: update user-defined cycle data with ambient variation
        self.ud_m_dot_water_cool_des = 0.0
        self.ud_ind_od = []

        self.tech_type = 1 # 1=fixed, 3=sliding
        self.P_boil = 100               # Boiler operating pressure [bar]

        self.CT = 2 # 1=evaporative, 2=air*, 3=hybrid
        self.T_amb_des = 42.0 #C
        self.P_cond_min = 2.0 #inHg
        return
    
    # Update parameters to best match CD data for a given simulated cycle type
    def initialize(self):
        self.set_solar_multiple()
        heliostat_layout = np.genfromtxt(self.heliostat_field_file, delimiter = ',')
        self.N_hel = heliostat_layout.shape[0]
        self.helio_positions = [heliostat_layout[j,0:2].tolist() for j in range(self.N_hel)]
        return
    
    def set_solar_multiple(self):
        self.solarm = self.Qrec / (self.P_ref / self.design_eff)
        return
    
    def get_cycle_thermal_rating(self):
        return self.P_ref / self.design_eff
    
    def get_number_of_heliostats(self):
        if self.N_hel == 0:
            self.initialize()
        return self.N_hel
    
    def get_cp_htf(self, TC):
        if self.rec_htf != 17:
            print ('HTF %d not recognized'%self.rec_htf)
            return 0.0
        TK = TC+273.15
        return (-1.0e-10*(TK**3) + 2.0e-7*(TK**2) + 5.0e-6*TK + 1.4387)*1000.  # J/kg/K
    
    def get_density_htf(self, TC):
        if self.rec_htf != 17:
            print ('HTF %d not recognized'%self.rec_htf)
            return 0.0
        TK = TC+273.15
        return -1.0e-7*(TK**3) + 2.0e-4*(TK**2) - 0.7875*TK + 2299.4  # kg/m3 
    
    def get_visc_htf(self, TC):
        if self.rec_htf != 17:
            print ('HTF %d not recognized'%self.rec_htf)
            return 0.0
        return max(1e-4, 0.02270616 - 1.199514e-4*TC + 2.279989e-7*TC*TC - 1.473302e-10*TC*TC*TC)
        
    def get_design_storage_mass(self):
        q_pb_design = self.get_cycle_thermal_rating()
        e_storage = q_pb_design * self.tshours * 1000.  # Storage capacity (kWht)
        cp = self.get_cp_htf(0.5*(self.T_htf_cold_des+self.T_htf_hot_des)) * 1.e-3  # kJ/kg/K
        m_storage = e_storage * 3600. / cp / (self.T_htf_hot_des - self.T_htf_cold_des)  # Active storage mass (kg)
        return m_storage
    

    def get_storage_mass(self):
        m_active = self.get_design_storage_mass()
        rho_avg = self.get_density_htf(0.5*(self.T_htf_cold_des+self.T_htf_hot_des))
        V_active = m_active / rho_avg
        V_tot = V_active / (1.0 - self.h_tank_min/self.h_tank)
        V_inactive = V_tot - V_active
        rho_hot_des = self.get_density_htf(self.T_htf_hot_des)
        rho_cold_des = self.get_density_htf(self.T_htf_cold_des)
        m_inactive_hot = V_inactive * rho_hot_des  # Inactive mass in hot storage (kg)
        m_inactive_cold = V_inactive * rho_cold_des  # Inactive mass in cold storage (kg)
        m_active_hot_max = V_active*rho_hot_des
        m_active_cold_max = V_active*rho_cold_des
        return m_active_hot_max, m_active_cold_max, m_inactive_hot, m_inactive_cold
    
    
    def get_cycle_design_mass_flow(self):
        q_des = self.get_cycle_thermal_rating()  # MWt
        cp_des = self.get_cp_htf(0.5*(self.T_htf_cold_des+self.T_htf_hot_des))  #J/kg/K
        m_des = q_des*1.e6 / (cp_des * (self.T_htf_hot_des - self.T_htf_cold_des))  # kg/s
        return m_des
    
    def get_receiver_design_mass_flow(self):
        cp_des = self.get_cp_htf(0.5*(self.T_htf_cold_des+self.T_htf_hot_des))  #J/kg/K
        m_des = self.Qrec*1.e6 / (cp_des * (self.T_htf_hot_des - self.T_htf_cold_des))  # kg/s
        return m_des







# Use this to store properties that are used in ssc, and (in some cases) also in the dispatch model
class PlantProperties:
    def __init__(self, is_testday=False):
        
        # Properties that are used in ssc and (in some cases) the dispatch model
        self.helio_optical_error_mrad = 1.53   # 2.625  # Heliostat total optical error (used in ssc as a slope error)
        self.helio_reflectance = 0.90        # Clean heliostat reflectivity
        self.rec_absorptance = 0.94           # Receiver solar absorptance
        self.epsilon = 0.88                   # Receiver IR emissivity
        self.hl_ffact = 1.0                   # Heat loss mutlipler in receiver code

        # Receiver operational properties and losses
        self.f_rec_min = 0.25 		      # Minimum receiver turndown ratio.
        self.csp_pt_rec_max_oper_frac = 1.2  # Maximum receiver mass flow rate fraction
        self.piping_loss = 10200	      # piping loss
        self.rec_tm_mult = 1.0            # Extra thermal mass in transient receiver model (mutliplier on combined HTF and tube thermal mass )
        self.f_T_htf_hot_des_is_hot_tank_in_min = 0.5  # TODO default value for this?
        
        
        # Receiver startup minimum time requirements for temperature-based startup model Note that these values won't be used unless the transient startup modeled is enabled
        self.min_preheat_time = 36./60
        self.min_fill_time = 10./60
        self.startup_ramp_time = 16./60
        
        # Receiver startup parameters for historical time/energy startup model.  Note that these values won't be used if the transient startup modeled is enabled
        self.rec_qf_delay = 0.25          # Energy-based receiver startup delay (fraction of rated thermal power).
        self.rec_su_delay = 0.20       # Fixed receiver startup delay time (hr). Set to historical median startup time derived from commands in CD HFCS log files
        
        # Heliostat tracking parasitics.  
        self.p_start = 0.025     # Heliostat field startup energy (per heliostat) (kWe-hr)
        self.p_track = 0.055   # Heliostat tracking power (per heliostat) (kWe)
        
        # Power cycle parasitics
        self.is_elec_heat_dur_off = True  # Use cycle electric heater parasitic when the cycle is off
        self.W_off_heat_frac = 0.05      # TODO SAM default for this? # Electric heaters when cycle is off state
        self.pb_pump_coef = 0.55          # HTF pumping power through power block (kWe / kg/s) (CD data)   [Accounts for Hot Pumps]
        self.pb_fixed_par = 0.0055        # Constant losses in system, includes ACC power (CD data) -- [Accounts for SGS Heat Trace, WCT Fans & Pumps, Cond Pump, SGS Recir Pumps, Compressors]
        self.aux_par = 0.023              # Aux heater, boiler parasitic - Off"
        self.bop_par = 0.0                # Balance of plant parasitic power fraction ----- [Accounts for Feedwater pump power]
        self.bop_par_f = 1                # Balance of plant parasitic power fraction - mult frac
        self.bop_par_0 = 0                # Balance of plant parasitic power fraction - const coeff
        self.bop_par_1 = 0.483            # Balance of plant parasitic power fraction - linear coeff
        self.bop_par_2 = 0                # Balance of plant parasitic power fraction - quadratic coeff

        # Power cycle startup and operational limits
        #TODO: ssc default cycle startup times seem short, can we derive these from the CD data?
        self.startup_frac = 0.5           # Fraction of design thermal power needed for cold startup
        self.startup_time = 0.5           # Time needed for power block startup (hr)
        self.cycle_cutoff_frac = 0.2      # Minimum cycle fraction.
        self.cycle_max_frac = 1.05
        self.q_sby_frac = 0.2             # standby fraction (0.2)


        #---------------------------------------------------------------------
        # Properties that are only used in the dispatch model, not ssc (note many, but not all, use the same names as in "dispatch.py")

        # Receiver
        self.q_rec_standby_fraction = 0.05       # Receiver standby energy consumption (fraction of design point thermal power)
        self.q_rec_shutdown_fraction = 0.0      # Receiver shutdown energy consumption (fraction of design point thermal power)
        self.rec_sd_delay = 0.0           # Note: not a true "minimum" requirement

        # Cycle
        self.startup_frac_warm = 0.1*0.333     # Fraction of design thermal power needed for warm startup (using 1/3 of cold startup fraction)

        self.Pc = [0.550]            # Slope of the HTF pumping power through the cycle regression region i [kWe / kg/s] (default: pb_pump_coef)
        self.Bc = [0.0]    # Intercept of the HTF pumping power through the cycle regression region i [kWe / kg/s]
        self.Pfw = [0.0]           # Slope of the feed water pumping power through the cycle regression region i [kWe / kg/s] (default zero, not a parameter in SAM)
        self.Bfw = [0.0]    # Intercept of the feed water pumping power through the cycle regression region i [kWe / kg/s]
        
        
        # TES
        self.mass_cs_min_frac = 0.02   # Minimum mass in cold tank (fraction of tank capacity). Note that this is just for the dispatch model and is different than the minimum mass stipulated by h_tank_min (e.g. inactive thermal mass) in ssc.
        self.mass_hs_min_frac = 0.02   # Minimum mass in hot tank (fraction of tank capacity). Note that this is just for the dispatch model and is different than the minimum mass stipulated by h_tank_min (e.g. inactive thermal mass) in ssc.
        self.min_soc_to_start_cycle = 0.3  #Ratio of TEX max in order to start cycle

        # Temperature limits 
        self.T_rout_min = 290.   # Minimum receiver HTF outlet T (C)
        self.T_rout_max = 574.   # Maximum receiver HTF outlet T (C)
        self.T_hs_min = 400.     # Minimum HTF T in hot storage (C)
        self.T_hs_max = 574.     # Maximum HTF T in hot storage (C)
        self.T_cs_min = 290.     # Minimum HTF T in cold storage (C)
        self.T_cs_max = 400.     # Maximum HTF T in cold storage (C)

        # Power cycle ramping limits and minimum up/down times
        # TODO: Need to define appropriate values (using non-binding limits for now).
        self.pc_ramp_start = 1.0         # Maximum fraction of design point power allowed when starting generation
        self.pc_ramp_shutdown = 1.0      # Maximum fraction of design point power allowed in time step before shutdown
        
        self.pc_rampup = 0.6         # Cycle max ramp up (fraction of max electrical output per hour)
        self.pc_rampdown = 12.          # Cycle max ramp down (fraction of max electrical output per hour)
        self.pc_rampup_vl = 1.          # Cyle max ramp up violation limit (fraction of max electrical output per hr)
        self.pc_rampdown_vl = 1.        # Cycle max ramp down violation limit (fraction of max electrical output per hr)
        self.Yu = 3.0                   # Cycle minmium up time (hr)
        self.Yd = 1.0 #8.0              # Cycle minmium up time (hr)
        
        # Operating costs
        self.Crec = 0.002       # Heliostat field and receiver operating cost ($/kWht)
        self.Crsu = 950         # Receiver cost penalty for cold startup ($/start) (ssc default = 950)
        self.Crhsp = 950/5      # Receiver cost penalty for hot startup ($/start)
        self.Cpc = 0.002        # Power cycle operating cost ($/kWhe)
        self.Ccsu = 6250       # Cycle cost penalty for cold startup ($/start)
        self.Cchsp = 6250/5    # Cycle cost penalty for hot startup ($/start)
        self.C_delta_w = 0.0    # Penalty for change in power cycle production (below violation limit) ($/kWe)
        self.C_v_w = 0.4        # Penalty for change in power cycle production (above violation limit) ($/kWe)
        self.Ccsb = 0.0         # Power cycle standby operating cost ($/kWhe)
        
        
        # Parasitics (in addition to those defined in ssc parameters above)
        #TODO: Note heliostat communication and cycle operating parasitics are not in ssc
        self.p_comm = 0.0416                 # Heliostat field communication parasitic (per heliostat) (kWe)  May need to add this parasitic to ssc.  
        self.Wht_fract = 500./669900         # Receiver heat trace energy consumption (kWe / MWt receiver thermal rating)
        self.Wht_fract_partload = 0.00026    # Receiver heat trace energy consumption when receiver is on (kWe / MWt receiver thermal rating) 
        self.Wc_fract = 0.0125               # Fixed cycle operating parasitic (fraction of cycle capacity)
        self.Wb_fract = 0.052                # Cycle standby operation parasitic load (fraction of cycle capacity)
        
        # 
        # TODO: is there a way of obtaining these from simulated data?
        self.alpha_b = -1.0
        self.alpha_T = 2.0
        self.alpha_m = 0.0
        self.beta_b = 0.1
        self.beta_T = -0.2
        self.beta_m = -2.0
        self.beta_mT = 3.0
        
        return
        
    




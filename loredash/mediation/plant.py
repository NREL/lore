import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from enum import Enum
import numpy as np
import datetime, rapidjson

try:
    from mediation import data_validator
except:
    from loredash.mediation import data_validator       # if running from case_study.py

#TODO: Implement these for 'rec_op_mode_initial'
class ReceiverState(Enum):
    OFF = 0
    STARTUP = 1
    ON = 2

#TODO: Implement these for 'pc_op_mode_initial'
class PowerCycleState(Enum):
    STARTUP = 0
    ON = 1
    STANDBY = 2
    OFF = 3
    STARTUP_CONTROLLED = 4

class Plant:
    def __init__(self, design, initial_state):
        self.design = {}
        self.flux_maps = {
            'A_sf_in':                              0.0,                                # m2 area of solar field
            'eta_map':                              [],                                 # efficiency map
            'flux_maps':                            [],                                 # flux maps
        }
        self.state = {
            # Field and receiver:
            'is_field_tracking_init':               False,                              # Is field tracking?
            'rec_op_mode_initial':                  0,                                  # Receiver operating mode
            'rec_startup_time_remain_init':         0.,                                 # Receiver startup time remaining (hr)
            'rec_startup_energy_remain_init':       0.,                                 # Receiver startup energy remaining (Wh)
            'disp_rec_persist0':                    0.,                                 # Time (hr) that receiver has been in its current state
            'disp_rec_off0':                        0.,                                 # Time (hr) that receiver has not been operating (off or startup)
            # TES:
            'T_tank_cold_init':                     0.,                                 # Cold tank temperature (C)
            'T_tank_hot_init':                      0.,                                 # Hot tank temperature (C)
            'csp_pt_tes_init_hot_htf_percent':      0.,                                 # Fraction of available storage in hot tank (%)
            # Power cycle:
            'pc_op_mode_initial':                   3,                                  # Initial cycle operating mode (0 = startup, 1 = on, 2 = standby, 3 = off, 4 = startup_controlled)
            'pc_startup_time_remain_init':          0.,                                 # Cycle startup time remaining (hr)
            'pc_startup_energy_remain_initial':     0.,                                 # Cycle startup energy remaining (kWh)
            'disp_pc_persist0':                     0.,                                 # Time (hr) that cycle has been in its current state
            'disp_pc_off0':                         0.,                                 # Time (hr) that cycle has not been generating electric power (off, startup, or standby)
            # TODO: these are cycle state variables?:
            'wdot0':                                0.,                                 # Cycle electricity generation (MWe)
            'qdot0':                                0.,                                 # Cycle thermal input (MWt)
        }

        self.set_design(design)
        self.set_initial_state(initial_state)

    def set_design(self, design):
        """
        Reads a plant design from either a JSON file or dict, validates the values, then adds them to the member design dict
        TODO: reinstitute the validation once the plant design is more finalized
        """
        if isinstance(design, str) and os.path.isfile(design):
            with open(design) as f:
                plant_design = rapidjson.load(f)                            # assume its a JSON file
        elif isinstance(design, dict):
            plant_design = design.copy()
        else:
            raise Exception('Plant design not found.')

        plant_design['N_hel'], plant_design['helio_positions'] = self.get_heliostat_field(plant_design['heliostat_field_file'])

        # validated_plant_design = data_validator.validate(plant_design, data_validator.plant_config_schema)
        self.design.update(plant_design)                                    # save plant design
        self.design['solarm'] = self.calc_solar_multiple(self.design)

    def set_flux_maps(self, flux_eta_maps):
        self.design['A_sf_in'] = flux_eta_maps['A_sf']
        self.design['eta_map'] = flux_eta_maps['eta_map_out']
        self.design['flux_maps'] = [x[2:] for x in flux_eta_maps['flux_maps_for_import']]
        return
    
    def set_initial_state(self, state):
        self.set_state(state)
        self.state['rec_startup_time_remain_init'] = self.design['rec_su_delay']
        self.state['rec_startup_energy_remain_init'] = self.design['rec_qf_delay'] * self.design['Qrec'] * 1.e6
        self.state['T_tank_cold_init'] = self.design['T_htf_cold_des']
        self.state['T_tank_hot_init'] = self.design['T_htf_hot_des']
        self.state['pc_startup_time_remain_init'] = self.design['startup_time']
        self.state['pc_startup_energy_remain_initial'] = self.design['startup_frac'] * self.get_cycle_thermal_rating() * 1000.

    def set_state(self, state):
        self.state.update((k, state[k]) for k in state.keys() & self.state.keys())     # update state but don't add any new keys

    def calc_solar_multiple(self, design):
        return design['Qrec'] / (design['P_ref'] / design['design_eff'])
    
    def get_heliostat_field(self, heliostat_field_file, delimiter=','):
        full_heliostat_field_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), heliostat_field_file)
        heliostat_layout = np.genfromtxt(full_heliostat_field_file_path, delimiter = delimiter)
        N_hel = heliostat_layout.shape[0]
        helio_positions = [heliostat_layout[j,0:2].tolist() for j in range(N_hel)]
        return N_hel, helio_positions



    def get_cycle_thermal_rating(self):
        return self.design['P_ref'] / self.design['design_eff']     # MW
    
    def get_number_of_heliostats(self):
        if not self.design['N_hel'] > 0:
            self.set_heliostat_field(self.design['heliostat_field_file'])
        return self.design['N_hel']
    
    def get_cp_htf(self, TC):
        if self.design['rec_htf'] != 17:
            print ('HTF %d not recognized'%self.design['rec_htf'])
            return 0.0
        TK = TC+273.15
        return (-1.0e-10*(TK**3) + 2.0e-7*(TK**2) + 5.0e-6*TK + 1.4387)*1000.  # J/kg/K
    
    def get_density_htf(self,TC):
        if self.design['rec_htf'] != 17:
            print ('HTF %d not recognized'%self.design['rec_htf'])
            return 0.0
        TK = TC+273.15
        return -1.0e-7*(TK**3) + 2.0e-4*(TK**2) - 0.7875*TK + 2299.4  # kg/m3 
    
    def get_visc_htf(self, TC):
        if self.design['rec_htf'] != 17:
            print ('HTF %d not recognized'%self.design['rec_htf'])
            return 0.0
        return max(1e-4, 0.02270616 - 1.199514e-4*TC + 2.279989e-7*TC*TC - 1.473302e-10*TC*TC*TC)
        
    def get_design_storage_mass(self):
        q_pb_design = self.get_cycle_thermal_rating()
        e_storage = q_pb_design * self.design['tshours'] * 1000.                                                # Storage capacity (kWht)
        cp = self.get_cp_htf(0.5*(self.design['T_htf_cold_des'] + self.design['T_htf_hot_des'])) * 1.e-3        # kJ/kg/K
        m_storage = e_storage * 3600. / cp / (self.design['T_htf_hot_des'] - self.design['T_htf_cold_des'])     # Active storage mass (kg)
        return m_storage
    
    def get_storage_mass(self):
        m_active = self.get_design_storage_mass()
        rho_avg = self.get_density_htf(0.5*(self.design['T_htf_cold_des'] + self.design['T_htf_hot_des']))
        V_active = m_active / rho_avg
        V_tot = V_active / (1.0 - self.design['h_tank_min'] / self.design['h_tank'])
        V_inactive = V_tot - V_active
        rho_hot_des = self.get_density_htf(self.design['T_htf_hot_des'])
        rho_cold_des = self.get_density_htf(self.design['T_htf_cold_des'])
        m_inactive_hot = V_inactive * rho_hot_des       # Inactive mass in hot storage (kg)
        m_inactive_cold = V_inactive * rho_cold_des     # Inactive mass in cold storage (kg)
        m_active_hot_max = V_active*rho_hot_des
        m_active_cold_max = V_active*rho_cold_des
        return m_active_hot_max, m_active_cold_max, m_inactive_hot, m_inactive_cold
    
    def get_cycle_design_mass_flow(self):
        q_des = self.get_cycle_thermal_rating()                                                             # MWt
        cp_des = self.get_cp_htf(0.5*(self.design['T_htf_cold_des'] + self.design['T_htf_hot_des']))        # J/kg/K
        m_des = q_des*1.e6 / (cp_des * (self.design['T_htf_hot_des'] - self.design['T_htf_cold_des']))      # kg/s
        return m_des
    
    def get_receiver_design_mass_flow(self):
        cp_des = self.get_cp_htf(0.5*(self.design['T_htf_cold_des'] + self.design['T_htf_hot_des']))                    # J/kg/K
        m_des = self.design['Qrec']*1.e6 / (cp_des * (self.design['T_htf_hot_des'] - self.design['T_htf_cold_des']))    # kg/s
        return m_des

    #TODO: Can set_design() and update_design() be merged?
    def update_design(self, design_vars):
        self.design.update(design_vars)         # update design (new keys can be added)

    #TODO: Are set_initial_state(), set_state() and update_state() all needed?
    def update_state(self, cycle_results, new_plant_state_vars, ssc_time_step):
        """
        Inputs:
            self.state                          previous plant state
                -rec_op_mode_initial
                -pc_op_mode_initial
                -disp_rec_persist0
                -disp_rec_off0
                -disp_pc_persist0
                -disp_pc_off0
            cycle_results                       subset of ssc solution
                -Q_thermal
                -q_startup
                -P_cycle
                -q_pb
                -q_dot_pc_startup
            new_plant_state_vars                gotten directly from the sim results
                -rec_op_mode_initial
                -pc_op_mode_initial
            ssc_time_step

        Outputs:
            self.state
        """
        # TODO: note that this doesn't consider subdivision of time steps

        def disp_rec_persist0():
            # Receiver state persistence disp_rec_persist0
            #  set the respective is_rec_current array values true if their state is the same as the final/current state
            previous_rec_state = self.state['rec_op_mode_initial']  # Receiver state before start of most recent set of simulation calls
            current_rec_state = new_plant_state_vars['rec_op_mode_initial']    # Receiver state at the end of the the most recent simulation call
            if current_rec_state== 2:     # On
                is_rec_current = np.array(cycle_results['Q_thermal']) > 1.e-3
            elif current_rec_state == 1:  # Startup
                is_rec_current = np.array(cycle_results['q_startup'])  > 1.e-3
            elif current_rec_state == 0:  # Off
                is_rec_current = (np.array(cycle_results['Q_thermal']) + np.array(cycle_results['q_startup'])) <= 1.e-3

            n = len(cycle_results['Q_thermal'])
            if n == 1 or np.abs(np.diff(is_rec_current)).max() == 0:  # Receiver did not change state over this simulation window:
                disp_rec_persist0 = n*ssc_time_step if previous_rec_state != current_rec_state else self.state['disp_rec_persist0'] + n*ssc_time_step
            else:
                i = np.where(np.abs(np.diff(is_rec_current)) == 1)[0][-1]
                disp_rec_persist0 = int(n-1-i)*ssc_time_step
            return disp_rec_persist0
        
        def disp_rec_off0():
            # Receiver state persistence disp_rec_off0
            current_rec_state = new_plant_state_vars['rec_op_mode_initial']    # Receiver state at the end of the the most recent simulation call
            is_rec_not_on = np.array(cycle_results['Q_thermal']) <= 1.e-3  # Array of time points receiver is not generating thermal power
            n = len(cycle_results['Q_thermal'])
            if current_rec_state == 2:  # Receiver is on
                disp_rec_off0 = 0.0
            elif is_rec_not_on.min() == 1:  # Receiver was off for the full simulated horizon
                disp_rec_off0 = self.state['disp_rec_off0'] + n*ssc_time_step  
            else: # Receiver shut off sometime during the current horizon
                i = np.where(np.abs(np.diff(is_rec_not_on)) == 1)[0][-1]
                disp_rec_off0 = int(n-1-i)*ssc_time_step
            return disp_rec_off0
            
        def disp_pc_persist0():
            # Cycle state persistence disp_pc_persist0
            previous_cycle_state = self.state['pc_op_mode_initial']   # Cycle state before start of most recent set of simulation calls
            current_cycle_state = new_plant_state_vars['pc_op_mode_initial']  # Cycle state at the end of the the most recent simulation call
            if current_cycle_state == 1: # On
                is_pc_current = np.array(cycle_results['P_cycle']) > 1.e-3 
            elif current_cycle_state == 2: # Standby
                is_pc_current = np.logical_and(np.logical_and(np.array(cycle_results['P_cycle'])<=1.e-3, np.array(cycle_results['q_pb'])>= 1.e-3), np.array(cycle_results['q_dot_pc_startup'])<=1.e-3)
            elif current_cycle_state == 0 or new_plant_state_vars['pc_op_mode_initial'] == 4: # Startup
                is_pc_current = np.array(cycle_results['q_dot_pc_startup']) > 1.e-3
            elif current_cycle_state == 3:  # Off
                is_pc_current = (np.array(cycle_results['q_dot_pc_startup']) + np.array(cycle_results['q_pb'])) <= 1.e-3

            n = len(cycle_results['P_cycle'])
            if n == 1 or np.abs(np.diff(is_pc_current)).max() == 0:  # Plant has not changed state over this simulation window:
                disp_pc_persist0 = n*ssc_time_step if previous_cycle_state != current_cycle_state else self.state['disp_pc_persist0'] + n*ssc_time_step
            else:
                i = np.where(np.abs(np.diff(is_pc_current)) == 1)[0][-1]
                disp_pc_persist0 = int(n-1-i)*ssc_time_step
            return disp_pc_persist0

        def disp_pc_off0():
            # Cycle state persistence disp_pc_off0
            current_cycle_state = new_plant_state_vars['pc_op_mode_initial']  # Cycle state at the end of the the most recent simulation call
            is_pc_not_on = np.array(cycle_results['P_cycle']) <=1.e-3
            n = len(cycle_results['P_cycle'])
            if current_cycle_state == 1:  # Cycle is on
                disp_pc_off0 = 0.0
            elif is_pc_not_on.min() == 1:  # Cycle was off for the full simulated horizon
                disp_pc_off0 = self.state['disp_pc_off0'] + n*ssc_time_step  
            else: # Cycle shut off sometime during the current horizon
                i = np.where(np.abs(np.diff(is_pc_not_on)) == 1)[0][-1]
                disp_pc_off0 = int(n-1-i)*ssc_time_step                        
            return disp_pc_off0

        new_persistance_vars = {
            'disp_rec_persist0': disp_rec_persist0(),
            'disp_rec_off0': disp_rec_off0(),
            'disp_pc_persist0': disp_pc_persist0(),
            'disp_pc_off0': disp_pc_off0()
            }

        self.state.update(new_persistance_vars)
        self.state.update(new_plant_state_vars)

        return

    def get_location(self):
        location = {
            'latitude':     self.design['latitude'],
            'longitude':    self.design['longitude'],
            'elevation':    self.design['elevation'],
            'timezone':     self.design['timezone'],
        }
        return location

    def set_location(self, location):
        self.design['latitude'] = location['latitude']
        self.design['longitude'] = location['longitude']
        self.design['elevation'] = location['elevation']
        self.design['timezone'] = location['timezone']


# Used just for case study
plant_initial_state_CD = {
    # Field and receiver:
    'is_field_tracking_init':               False,                              # Is field tracking?
    'rec_op_mode_initial':                  0,                                  # Receiver operating mode
    'rec_startup_time_remain_init':         0.,                                 # Receiver startup time remaining (hr)
    'rec_startup_energy_remain_init':       0.,                                 # Receiver startup energy remaining (Wh)
    'disp_rec_persist0':                    1000.,                              # Time (hr) that receiver has been in its current state
    'disp_rec_off0':                        1000.,                              # Time (hr) that receiver has not been operating (off or startup)
    # TES:
    'T_tank_cold_init':                     0.,                                 # Cold tank temperature (C)
    'T_tank_hot_init':                      0.,                                 # Hot tank temperature (C)
    'csp_pt_tes_init_hot_htf_percent':      30.,                                # Fraction of available storage in hot tank (%)
    # Power cycle:
    'pc_op_mode_initial':                   3,                                  # Initial cycle operating mode (0 = startup, 1 = on, 2 = standby, 3 = off, 4 = startup_controlled)
    'pc_startup_time_remain_init':          0.,                                 # Cycle startup time remaining (hr)
    'pc_startup_energy_remain_initial':     0.,                                 # Cycle startup energy remaining (kWh)
    'disp_pc_persist0':                     1000.,                              # Time (hr) that cycle has been in its current state
    'disp_pc_off0':                         1000.,                              # Time (hr) that cycle has not been generating electric power (off, startup, or standby)
    # TODO: these are cycle state variables?:
    'wdot0':                                0.,                                 # Cycle electricity generation (MWe)
    'qdot0':                                0.,                                 # Cycle thermal input (MWt)
}


# NOTE: These are the values of the corresponding outputs after 5 minutes of operation,
# starting with the PySAM default initialization
plant_initial_state = {
    # Field and receiver:
    'is_field_tracking_init':               0,                                  # Is field tracking?
    'rec_op_mode_initial':                  0,                                  # Receiver operating mode
    'rec_startup_time_remain_init':         0.2,                                # Receiver startup time remaining (hr)
    'rec_startup_energy_remain_init':       167475728,                          # Receiver startup energy remaining (Wh)
    'disp_rec_persist0':                    1000.,                              # Time (hr) that receiver has been in its current state
    'disp_rec_off0':                        1000.,                              # Time (hr) that receiver has not been operating (off or startup)
    # TES:
    'T_tank_cold_init':                     290.,                               # Cold tank temperature (C)
    'T_tank_hot_init':                      573.9,                              # Hot tank temperature (C)
    'csp_pt_tes_init_hot_htf_percent':      25.0,                               # Fraction of available storage in hot tank (%)
    # Power cycle:
    'pc_op_mode_initial':                   1,                                  # Initial cycle operating mode (0 = startup, 1 = on, 2 = standby, 3 = off, 4 = startup_controlled)
    'pc_startup_time_remain_init':          0.,                                 # Cycle startup time remaining (hr)
    'pc_startup_energy_remain_initial':     0.,                                 # Cycle startup energy remaining (kWh)
    'disp_pc_persist0':                     1000.,                              # Time (hr) that cycle has been in its current state
    'disp_pc_off0':                         1000.,                              # Time (hr) that cycle has not been generating electric power (off, startup, or standby)
    # TODO: these are cycle state variables?:
    'wdot0':                                0.,                                 # Cycle electricity generation (MWe)
    'qdot0':                                0.,                                 # Cycle thermal input (MWt)
}


plant_design = {
    'name':                         "Crescent Dunes",

    # Location
    "latitude":                     38.23895540732931,
    "longitude":                    -117.36368180656123,
    "elevation":                    1524,
    "timezone":                     -8,

    # System
    'T_htf_cold_des':               295.0,          # C
    'T_htf_hot_des':                565.0,          # C
    'P_ref':                        120,            # MW
    'gross_net_conversion_factor':  0.92,
    'design_eff':                   0.409,
    'Qrec':                         565.0,          # MWt
    'tshours':                      10.1,           # hr (10.1 results in a close match of total salt mass compared to CD data)(TODO: using 10 hrs, SSC hangs during simulation??)  
    'solarm':                       None,           # = self.Qrec * self.design_eff / self.P_ref

    # Solar field
    'heliostat_field_file':         '../../librtdispatch/input_files/CD_processed/Crescent_Dunes_heliostat_layout.csv',   # TODO: This the field layout with a fixed z-coordinate.
                                                                                                        # Could update to call through SolarPILOT API with actual z-coordinates?
                                                                                                        # (but model validation cases suggest this is not terribly important)
    'helio_positions':              [],
    'N_hel':                        0,
    'helio_height':                 11.28,          # m
    'helio_width':                  10.36,          # m
    'n_facet_x':                    7,
    'n_facet_y':                    5,
    'dens_mirror':                  0.97,
    'helio_optical_error_mrad':     2.5,            # 2.625  # Heliostat total optical error (used in ssc as a slope error)
    'helio_reflectance':            0.943,          # Clean heliostat reflectivity
    'p_start':                      0.0,            # TODO: Heliostat field startup energy (per heliostat) (kWe-hr) Currently using SAM defaults, update based on CD data?
    'p_track':                      0.0477,         # Heliostat tracking power (per heliostat) (kWe)
    # Adding these here will change the results:
    # 'A_sf_in':                      0.0,            # m2 area of solar field, used with flux maps
    # 'eta_map':                      [],             # efficiency map, used with flux maps
    # 'flux_maps':                    [],             # flux maps

    # Receiver
    # --TODO: Find a meaningful value for energy requirement (default is 0.25)... set to a low value so that time requirement is binding?
    # --TODO: Update transient model for frequent simulations starts and stops. SS model requires min turndown ratio for all of startup, most starting time is actually in preheat. 
    'rec_htf':                      17,             # 17 = 60% NaNO3, 40% KNO3 nitrate salt
    'rec_height':                   18.59,          # Heated panel length (m)
    'D_rec':                        15.18,          # m
    'd_tube_out':                   50.8,           # mm
    'th_tube':                      1.245,          # mm
    'N_panels':                     14,
    'h_tower':                      175,            # m
    'mat_tube':                     32,             # 2 = SS316, 32 = H230, 33 = 740H
    'piping_length_mult':           2.6,            # TODO: Better value for CD?  Piping length multiplier
    'Flow_type':                    1,
    'crossover_shift':              -1,             # Shift flow crossover to match CD (3 panels before cross, 4 panels after).
                                                    # Note that ssc flow path designations are opposite of CD.
                                                    # In ssc path 1 starts northwest and crosses to southeast, in CD path 1 starts northeast and crosses to southwest.
    'eta_pump':                     0.52,           # Receiver pump efficiency, set to approx. match pump parasitic from CD data (peak power ~5.25 MWe) but this seems like a low efficiency
    'header_sizing':                [609.6, 2.54, 3.353, 32],   # Header sizing ([OD(mm), wall (mm), length (m), material])
    'crossover_header_sizing':      [406.4, 12.7, 30.18, 2],    # Crossover header sizing ([OD(mm), wall (mm), length (m), material])  
    'rec_absorptance':              0.94,           # Receiver solar absorptance
    'epsilon':                      0.88,           # Receiver IR emissivity
    'hl_ffact':                     1.0,            # Heat loss mutlipler in receiver code
    'q_rec_standby_fraction':       0.05,           # TODO: Receiver standby energy consumption (fraction of design point thermal power)
    'q_rec_shutdown_fraction':      0.0,            # TODO: Receiver shutdown energy consumption (fraction of design point thermal power)
    'rec_sd_delay':                 0.0,            # TODO: Based on median post-heat time from CD data, but not a true "minimum" requirement
    'T_rout_min':                   280.,           # Minimum receiver HTF outlet T (C)
    'T_rout_max':                   565.,           # Maximum receiver HTF outlet T (C)
    # --operational properties and losses
    'f_rec_min':                    0.25, 	        # TODO: Minimum receiver turndown ratio.  Using ssc default for now, but maybe we can derive from CD data
    'csp_pt_rec_max_oper_frac':     1.2,            # TODO: Maximum receiver mass flow rate fraction .  Using ssc default for now, but maybe we can derive from CD data 
    'piping_loss':                  0.0, 	        # TODO: Need a better estimate of piping loss, was set to zero for model validation cases as outlet T measurements were before downcomer
    'rec_tm_mult':                  1.0,            # Extra thermal mass in transient receiver model (mutliplier on combined HTF and tube thermal mass )
    'f_T_htf_hot_des_is_hot_tank_in_min':0.5,       # Temperature threshold for rec output to cold tank = f*T_hot_des + (1-f)*T_cold_des.
                                                    # Temperature threshold should be ~425C based on CD discussions. This will always trigger cold tank recirculation in ssc (regardless of dispatch signal).
    # --startup minimum time requirements for temperature-based startup model (from median times based on CD data). Note that these values won't be used unless the transient startup modeled is enabled
    'min_preheat_time':             36./60,         # Historical median Preheat time derived from commands in CD HFCS log files
    'min_fill_time':                10./60,         # Median fill time derived from commands in CD HFCS log files
    'startup_ramp_time':            16./60,         # Combined historical median Operate and Track (to near 100% or point in time tracking stops changing)
    # --startup parameters for historical time/energy startup model.  Note that these values won't be used if the transient startup modeled is enabled
    'rec_qf_delay':                 0.25,           # Energy-based receiver startup delay (fraction of rated thermal power).
    'rec_su_delay':                 69./60,         # Fixed receiver startup delay time (hr). Set to historical median startup time derived from commands in CD HFCS log files

    # TES
    'h_tank':                       11.1,           # m  # Tank height (m)
    'h_tank_min':                   1.0,            # m  # Minimum allowable HTF height in storage tank (m)
    'hot_tank_Thtr':                450,            # C
    'cold_tank_Thtr':               250,            # C       
    'mass_cs_min_frac':             0.02,           # Minimum mass in cold tank (fraction of tank capacity).
                                                    # Note that this is just for the dispatch model and is different than the minimum mass stipulated by h_tank_min (e.g. inactive thermal mass) in ssc.
    'mass_hs_min_frac':             0.02,           # Minimum mass in hot tank (fraction of tank capacity).
                                                    # Note that this is just for the dispatch model and is different than the minimum mass stipulated by h_tank_min (e.g. inactive thermal mass) in ssc.
    'min_soc_to_start_cycle':       0.3,            # Ratio of TEX max in order to start cycle
    'T_hs_min':                     400.,           # Minimum HTF T in hot storage (C)
    'T_hs_max':                     565.,           # Maximum HTF T in hot storage (C)
    'T_cs_min':                     280.,           # Minimum HTF T in cold storage (C)
    'T_cs_max':                     400.,           # Maximum HTF T in cold storage (C)

    # Power cyle
    # --TODO: ssc default cycle startup times seem short, can we derive these from the CD data?
    # --TODO: Based on CD data... generalize these from plant design parameters? 
    # --TODO: Make sure these are consistent with ssc parameter for cycle HTF pumping parasitic
    # --TODO: Need to define appropriate ramping limits and min up/down time values (using non-binding limits for now).  Or could initialize these directly in the dispatch model
    'pc_config':                    0,              # 0=Steam Rankine, 1=User-defined
    'ud_f_W_dot_cool_des':          0.58333,        # This is an estimate for the cooling system (700 kW) TODO: update user-defined cycle data with ambient variation
    'ud_m_dot_water_cool_des':      0.0,
    'ud_ind_od':                    [[0]],
    'tech_type':                    1,              # 1=fixed, 3=sliding
    'P_boil':                       125,            # Boiler operating pressure [bar]
    'CT':                           2,              # 1=evaporative, 2=air*, 3=hybrid
    'T_amb_des':                    42.8,           #58   # C
    'P_cond_min':                   1.0,            #3.0  # inHg
    'is_elec_heat_dur_off':         True,           # Use cycle electric heater parasitic when the cycle is off
    'W_off_heat_frac':              0.045,          # Electric heaters when cycle is off state (CD data) [Accounts for Startup and Aux Boiler Electric Heaters (~5.4 MWe)]
    'pb_pump_coef':                 0.86,           # HTF pumping power through power block (kWe / kg/s) (CD data)   [Accounts for Hot Pumps]  
    'pb_fixed_par':                 0.010208,       # Constant losses in system, includes ACC power (CD data) -- [Accounts for SGS Heat Trace, WCT Fans & Pumps, Cond Pump, SGS Recir Pumps, Compressors]
    'aux_par':                      0.0,            # Aux heater, boiler parasitic - Off"           
    'bop_par':                      0.008,          # Balance of plant parasitic power fraction ----- [Accounts for Feedwater pump power]
    'bop_par_f':                    1,              # Balance of plant parasitic power fraction - mult frac
    'bop_par_0':                    1,              # Balance of plant parasitic power fraction - const coeff
    'bop_par_1':                    -0.51530293,    # Balance of plant parasitic power fraction - linear coeff
    'bop_par_2':                    1.92647426,     # Balance of plant parasitic power fraction - quadratic coeff
    'startup_frac':                 0.1,            # Fraction of design thermal power needed for cold startup
    'startup_time':                 0.5,            # Time needed for power block startup (hr)
    'cycle_cutoff_frac':            0.3,            # TODO: Minimum cycle fraction. Updated from 0.1 used in model validation to 0.3 (minimum point in user-defined pc file)
    'cycle_max_frac':               1.0,        
    'q_sby_frac':                   0.2,            # TODO: using ssc default standby fraction (0.2), update from CD data?
    'startup_frac_warm':            0.1*0.333,      # Fraction of design thermal power needed for warm startup (using 1/3 of cold startup fraction)
    'Pc':                           [0.560518, 0.876873, 1.272042],             # Slope of the HTF pumping power through the cycle regression region i [kWe / kg/s]
    'Bc':                           [-31.142492, -149.254350, -361.970846],     # Intercept of the HTF pumping power through the cycle regression region i [kWe / kg/s]
    'Pfw':                          [1.231783, 2.511473, 4.108881],             # Slope of the feed water pumping power through the cycle regression region i [kWe / kg/s]
    'Bfw':                          [605.778892, 127.911967, -732.148085],      # Intercept of the feed water pumping power through the cycle regression region i [kWe / kg/s] 
    # --ramping limits and minimum up/down times
    'pc_ramp_start':                1.0,            # Maximum fraction of design point power allowed when starting generation
    'pc_ramp_shutdown':             1.0,            # Maximum fraction of design point power allowed in time step before shutdown
    'pc_rampup':                    0.6,            # Cycle max ramp up (fraction of max electrical output per hour)
    'pc_rampdown':                  12.,            # Cycle max ramp down (fraction of max electrical output per hour)
    'pc_rampup_vl':                 1.,             # Cyle max ramp up violation limit (fraction of max electrical output per hr)
    'pc_rampdown_vl':               1.,             # Cycle max ramp down violation limit (fraction of max electrical output per hr)
    'Yu':                           3.0,            # Cycle minmium up time (hr)
    'Yd':                           1.0,            # Cycle minmium up time (hr)

    # Operating costs
    # --TODO (low): Most use ssc defaults and  values currently being using in the nonlinear dispatch model.  Probably ok, but might be worth revisiting as needed
    'Crec':                         0.002,          # Heliostat field and receiver operating cost ($/kWht)
    'Crsu':                         950,            # Receiver cost penalty for cold startup ($/start) (ssc default = 950)
    'Crhsp':                        950/5,          # Receiver cost penalty for hot startup ($/start)
    'Cpc':                          0.002,          # Power cycle operating cost ($/kWhe)
    'Ccsu':                         6250,           # Cycle cost penalty for cold startup ($/start)
    'Cchsp':                        6250/5,         # Cycle cost penalty for hot startup ($/start)
    'C_delta_w':                    0.0,            # Penalty for change in power cycle production (below violation limit) ($/kWe)
    'C_v_w':                        0.4,            # Penalty for change in power cycle production (above violation limit) ($/kWe)
    'Ccsb':                         0.0,            # Power cycle standby operating cost ($/kWhe)

    # Parasitics (in addition to those defined in ssc parameters above)
    # --TODO: Note heliostat communication and cycle operating parasitics are not in ssc
    'p_comm':                       0.0416,         # Heliostat field communication parasitic (per heliostat) (kWe)  May need to add this parasitic to ssc.  
    'Wht_fract':                    0.00163,        # Receiver heat trace energy consumption (kWe / MWt receiver thermal rating) 
    'Wht_fract_partload':           0.00026,        # Receiver heat trace energy consumption when receiver is on (kWe / MWt receiver thermal rating) 
    'Wc_fract':                     0.0125,         # Fixed cycle operating parasitic (fraction of cycle capacity)
    'Wb_fract':                     0.052,          # Cycle standby operation parasitic load (fraction of cycle capacity)

    # CD specific data?
    # --TODO: Are these specific from CD data? Do we need to generalize to calculate from anything related to cycle model or plant design?
    'alpha_b':                      -1.23427118048079,
    'alpha_T':                      2.24251108579222,
    'alpha_m':                      -0.0317161796408163,
    'beta_b':                       0.0835548456010428,
    'beta_T':                       -0.176758581118889,
    'beta_m':                       -1.76744346976718,
    'beta_mT':                      2.84782141640903
}


#TODO: Implement these more ideal names
# plant_initial_state = {
#     # Field and receiver:
#     'field_is_tracking':                    False,                              # Is field tracking?
#     'receiver_state':                       0,                                  # Receiver operating mode
#     'receiver_startup_duration_remaining':  0.,                                 # Receiver startup time remaining (hr)
#     'receiver_startup_energy_remaining':    0.,                                 # Receiver startup energy remaining (Wh)
#     'receiver_current_state_duration':      1000.,                              # Time (hr) that receiver has been in its current state
#     'receiver_duration_not_on':             1000.,                              # Time (hr) that receiver has not been operating (off or startup)
#     # TES:
#     'T_tes_tank_cold':                      0.,                                 # Cold tank temperature (C)
#     'T_tes_tank_hot':                       0.,                                 # Hot tank temperature (C)
#     'Fill_fraction_hot_tank':               30.,                                # Fraction of available storage in hot tank (%)
#     # Power cycle:
#     'cycle_state':                          3,                                  # Initial cycle operating mode (0 = startup, 1 = on, 2 = standby, 3 = off, 4 = startup_controlled)
#     'cycle_startup_duration_remaining':     0.,                                 # Cycle startup time remaining (hr)
#     'cycle_startup_energy_remaining':       0.,                                 # Cycle startup energy remaining (kWh)
#     'cycle_current_state_duration':         1000.,                              # Time (hr) that cycle has been in its current state
#     'cycle_duration_not_on':                1000.,                              # Time (hr) that cycle has not been generating electric power (off, startup, or standby)
#     # TODO: these are cycle state variables?:
#     'W_cycle':                              0.,                                 # Cycle electricity generation (MWe)
#     'Q_cycle':                              0.,                                 # Cycle thermal input (MWt)
# }

#TODO: Implement this with the above
# plant_state_to_ssc_inputs = {    
#     # Field and receiver
#     'field_is_tracking':                    'is_field_tracking_init',           # Is field tracking?
#     'receiver_state':                       'rec_op_mode_initial',              # Receiver operating mode (0 = off, 1 = startup, 2 = on)
#     'receiver_startup_duration_remaining':  'rec_startup_time_remain_init',     # Receiver startup time remaining (hr)
#     'receiver_startup_energy_remaining':    'rec_startup_energy_remain_init',   # Receiver startup energy remaining (Wh)
#     'receiver_current_state_duration':      'disp_rec_persist0',                # Time (hr) that receiver has been in its current state
#     'receiver_duration_not_on':             'disp_rec_off0',                    # Time (hr) that receiver has not been operating (off or startup)
    
#     # TES
#     'T_tes_tank_cold':                      'T_tank_cold_init',                 # Cold tank temperature (C)
#     'T_tes_tank_hot':                       'T_tank_hot_init',                  # Hot tank temperature (C)
#     'Fill_fraction_hot_tank':               'csp_pt_tes_init_hot_htf_percent',  # Fraction of available storage in hot tank (%)
    
#     # Power cycle
#     'cycle_state':                          'pc_op_mode_initial',               # Initial cycle operating mode (0 = startup, 1 = on, 2 = standby, 3 = off, 4 = startup_controlled)
#     'cycle_startup_duration_remaining':     'pc_startup_time_remain_init',      # Cycle startup time remaining (hr)
#     'cycle_startup_energy_remaining':       'pc_startup_energy_remain_initial', # Cycle startup energy remaining (kWh)
#     'cycle_current_state_duration':         'disp_pc_persist0',                 # Time (hr) that cycle has been in its current state
#     'cycle_duration_not_on':                'disp_pc_off0',                     # Time (hr) that cycle has not been generating electric power (off, startup, or standby)
#     # TODO: these are state variables?:
#     'W_cycle':                              'wdot0',                            # Cycle electricity generation (MWe)
#     'Q_cycle':                              'qdot0'                             # Cycle thermal input (MWt)
# }


# This should really be in mediator (probably), but is here for the sake of case_study
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
        time_of_year = (start - datetime.datetime(start.year,1,1,0,0,0)).total_seconds()
        startpt = int(time_of_year/3600) * nph   # First point in annual arrays         
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
    def calculate_day_ahead_penalty(sim_days, schedules, P_out_net, params, disp_soln_tracking=[], 
        disp_params_tracking=[], disp_net_electrical_output=None):
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

    
    @staticmethod
    def get_price_data(price_multiplier_file, avg_price, price_steps_per_hour, time_steps_per_hour):

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

        price_multipliers = np.genfromtxt(os.path.join(os.path.dirname(__file__), price_multiplier_file))
        if price_steps_per_hour != time_steps_per_hour:
            price_multipliers = translate_to_new_timestep(price_multipliers, 1./price_steps_per_hour, 1./time_steps_per_hour)
        pmavg = sum(price_multipliers)/len(price_multipliers)  
        price_data = [avg_price*p/pmavg  for p in price_multipliers]  # Electricity price at ssc time steps ($/MWh)
        return price_data

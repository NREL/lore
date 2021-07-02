# We need access to the librtdispatch directory. However, don't just assume that
# the first entry in the path is `loredash`, because this file may get called
# from a Bokeh app.
import os, sys
dir_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), # /mediation
    '..',                                        # /loredash
    '..',                                        # /lore
)
sys.path.insert(1, dir_path)
from enum import Enum
import numpy as np
import pandas as pd
import datetime, rapidjson

from mediation import data_validator

class Plant:
    """Represents a real plant. Includes attributes that characterize the plant or
    those used in preprocessing for calculating them. Also includes methods to determine
    the real plant state.
    """

    # TODO: do these have everything needed from ssc?
    def __init__(self, design, initial_state):
        self.design = {}
        self.flux_maps = {
            'A_sf_in':                              0.0,                                # m2 area of solar field
            'eta_map':                              [],                                 # efficiency map
            'flux_maps':                            [],                                 # flux maps
        }
        self.state = {
            """This is the current plant state, but it's usually only updated at the beginning
            and end of a timestep. The names correspond to SSC inputs (hence the 'init's and 0's).
            """
            # Field and receiver:
            'is_field_tracking_init':               False,                              # Is field tracking?
            'rec_op_mode_initial':                  0,                                  # Receiver operating mode
            'rec_startup_time_remain_init':         0.,                                 # Receiver startup time remaining (hr)
            'rec_startup_energy_remain_init':       0.,                                 # Receiver startup energy remaining (Wh)
            'disp_rec_persist0':                    0.,                                 # Time (hr) that receiver has been in its current state
            'disp_rec_off0':                        0.,                                 # Time (hr) that receiver has not been operating (off or startup)
            'sf_adjust:hourly':                     0.,                                 # Solar field adjustment factor (latest)
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
        self.state['sf_adjust:hourly'] = self.get_field_availability()[-1]

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

    def get_state(self):
        result = self.state.copy()              # copy to disallow edits
        result['sf_adjust:hourly'] = self.get_field_availability()[-1]
        return result

    def get_design(self):
        return self.design.copy()

    def get_field_availability(self, datetime_start=None, duration=None, timestep=None):
        #TODO: replace this function body with call to real plant
        """output array must be equal in length to the weather data to satisfy ssc"""
        if datetime_start is None:
            datetime_start = datetime.datetime(2018, 1, 1)
        if duration is None:
            duration = datetime.timedelta(days=365)
        if timestep is None:
            timestep = datetime.timedelta(minutes=1)

        FIELD_AVAIL_DAYS_GENERATED = 365
        steps_per_hour = int(1/(timestep.total_seconds()/3600))

        fixed_soiling_loss = 0.02       # TODO: move this constant to the plant config file
        field_availability = (fixed_soiling_loss * 100 * np.ones(steps_per_hour*24*365)).tolist()  

        assert(len(field_availability) == steps_per_hour * 24 * FIELD_AVAIL_DAYS_GENERATED)
        df = pd.DataFrame(field_availability, columns=['field_availability'])
        df.index = pd.date_range(start=datetime_start,
                                 end=datetime_start + datetime.timedelta(days=FIELD_AVAIL_DAYS_GENERATED) - timestep,
                                 freq=timestep)

        df_out = df[datetime_start:(datetime_start + duration - timestep)]
        return list(df_out['field_availability'])


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

    def update_flux_maps(self, flux_eta_maps):
        self.flux_maps.update(flux_eta_maps)

    #TODO: Are set_initial_state() and set_state() both needed?
    def calc_persistance_vars(self, cycle_results, ssc_time_step):
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
            ssc_time_step

        Outputs:
            self.state
        """
        # TODO: note that this doesn't consider subdivision of time steps

        def disp_rec_persist0():
            # Receiver state persistence disp_rec_persist0
            #  set the respective is_rec_current array values true if their state is the same as the final/current state
            previous_rec_state = self.state['rec_op_mode_initial']  # Receiver state before start of most recent set of simulation calls
            current_rec_state = cycle_results['rec_op_mode_final'][-1]    # Receiver state at the end of the the most recent simulation call
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
            current_rec_state = cycle_results['rec_op_mode_final'][-1]    # Receiver state at the end of the the most recent simulation call
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
            current_cycle_state = cycle_results['pc_op_mode_final'][-1]  # Cycle state at the end of the the most recent simulation call
            if current_cycle_state == 1: # On
                is_pc_current = np.array(cycle_results['P_cycle']) > 1.e-3 
            elif current_cycle_state == 2: # Standby
                is_pc_current = np.logical_and(np.logical_and(np.array(cycle_results['P_cycle'])<=1.e-3, np.array(cycle_results['q_pb'])>= 1.e-3), np.array(cycle_results['q_dot_pc_startup'])<=1.e-3)
            elif current_cycle_state == 0 or cycle_results['pc_op_mode_final'][-1] == 4: # Startup
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
            current_cycle_state = cycle_results['pc_op_mode_final'][-1]  # Cycle state at the end of the the most recent simulation call
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

        return new_persistance_vars

    def get_location(self):
        location = {
            'latitude':         self.design['latitude'],
            'longitude':        self.design['longitude'],
            'elevation':        self.design['elevation'],
            'timezone':         self.design['timezone'],
            'timezone_string':  self.design['timezone_string'],
        }
        return location

    def set_location(self, location):
        self.design['latitude'] = location['latitude']
        self.design['longitude'] = location['longitude']
        self.design['elevation'] = location['elevation']
        self.design['timezone'] = location['timezone']


# NOTE: These are the values of the corresponding outputs after 5 minutes of operation,
# starting with the ssc default initialization
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

# TODO: move this out of plant.py . Maybe to a new financials.py file?
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

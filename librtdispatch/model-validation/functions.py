import os
import pickle
import pandas as pd
from datetime import datetime
import datetime
import matplotlib.pyplot as plt

import numpy as np
import copy
import PySSC as api
from mspt_2020_defaults import vartab as V

import timeit


class design_parameters:
    def __init__(self):
        self.P_ref = 120  # MW
        self.gross_net_conversion_factor = 0.92
        self.design_eff = 0.385 #0.412 # 0.4285 # NOTE: Adjusted to match gross cycle output
        self.Qrec = 565.0  # MWt
        self.tshours = 11.5 #10.9** # hr  (Actual size is 10 hour, can simulate with large storage size if just looking at receiver) NOTE: Adjusted to match total salt mass 
        self.solarm = self.Qrec * self.design_eff / self.P_ref

        self.pc_config = 0 # 0=Steam Rankine, 1=User-defined
        self.ud_f_W_dot_cool_des = 0.0  # TODO: update these?
        self.ud_m_dot_water_cool_des = 0.0
        self.ud_ind_od = []

        self.tech_type = 1 # 1=fixed, 3=sliding
        self.cycle_max_frac = 1.0       # Maximum turbine over design operation fraction
        self.P_boil = 125               # Boiler operating pressure [bar]

        self.CT = 2 # 1=evaporative, 2=air*, 3=hybrid
        self.T_amb_des = 42.8 #58   # C
        self.P_cond_min = 1.0 #3.0  # inHg
        
        self.T_htf_cold_des = 290.0 # C
        self.T_htf_hot_des = 565.0 # C
        self.h_hank = 12.5# 12.7 #11.16 # m
        self.h_tank_min = 0.0 #1.075 #1.08 # m
        self.hot_tank_Thtr = 450 # C
        self.cold_tank_Thtr = 250 # C
    
        self.helio_height = 11.28 # m
        self.helio_width = 10.36 # m
        self.n_facet_x = 7
        self.n_facet_y = 5
        self.dens_mirror = 0.97
        
        self.rec_height = 18.59     # Heated panel length (m)
        self.D_rec = 15.18  # m
        self.d_tube_out = 50.8 # mm
        self.th_tube = 1.245 # mm
        self.N_panels = 14
        self.h_tower = 175 # m
        self.mat_tube = 32   # 2 = SS316, 32 = H230, 33 = 740H
        
        self.crossover_shift = -1  # Shift flow crossover to match CD (3 panels before cross, 4 panels after).  Note that ssc flow path designations are opposite of CD. In ssc path 1 starts northwest and crosses to southeast, in CD path 1 starts northeast and crosses to southwest.
        self.header_sizing = [609.6, 2.54, 3.353, 32]            # Header sizing ([OD(mm), wall (mm), length (m), material])
        self.crossover_header_sizing = [406.4, 12.7, 30.18, 2]   # Crossover header sizing ([OD(mm), wall (mm), length (m), material])
        return


class properties:
    def __init__(self, is_testday=False):
        self.reflectivity = 'measured'    # Heliostat reflectivity: numerical value (0-1), 'measured' = average value from CD data on specified day, 'plus_2sigma' or 'minus_2sigma' = 2 standard deviations from mean value
        
        self.helio_optical_error_mrad = 2.75  # Heliostat total optical error (used in ssc as a slope error)
        self.helio_reflectance = 0.943        # Clean heliostat reflectivity
        self.rec_absorptance = 0.94           # Receiver solar absorptance
        self.epsilon = 0.88                   # Receiver IR emissivity
        self.hl_ffact = 1.0                   # Heat loss mutlipler in receiver code

        # Receiver operational properties and losses
        self.f_rec_min = 0.00001 		# Minimum receiver turndown ratio (Small min turndown allows approximate simulation of standby periods, default ssc value is 0.25)
        self.piping_length_mult = 2.6   # Piping length multiplier (using small value to avoid simulating riser/downcomer piping, CD data is at receiver inlet/outlet)
        self.piping_loss = 0.0 		    # Piping loss per m (using zero here as CD inlet/outlet T are at receiver, ssc inlet/outlet T are at riser/downcomer). ssc default is 10200 W/m
        self.rec_tm_mult = 1.0          # Extra thermal mass in transient receiver model (mutliplier on combined HTF and tube thermal mass )

        # Minimum startup time requirements (from median times based on CD data). Note that these values won't be used unless the transient startup modeled is enabled
        self.min_preheat_time = 0.48    # Combined historical median STH-Fill, STH-Circulate, Preheat time from CD data
        self.min_fill_time = 0.17		# Median fill time from CD data
        self.startup_ramp_time = 0.29   # Combined historical median Operate and Track (to near 100% or point in time tracking stops changing)

        # Power cycle parasitics
        self.W_off_heat_frac = 0.034*2    # Electric heaters when cycle is off state (CD data)
        self.pb_pump_coef = 1.1*4         # HTF pumping power through power block (CD data)   *4 to include FW and cond. pumps   
        self.pb_fixed_par = 0.015       # Constant losses in system, includes ACC power (CD data)

        self.startup_frac = 0.1         # Fraction of design thermal power needed for startup
        self.startup_time = 0.5         # Time needed for power block startup
        self.cycle_cutoff_frac = 0.1

        if is_testday:
            self.n_flux_days = 2
            self.delta_flux_hrs = 4

    def set_dispatch_targets(self, cd, des_params,  use_avg_flow = False, controller_adj = False):
        self.is_dispatch_targets = True

        ## determine cycle mode
        if cd['Gross Power [MW]'][0] > 0:
            self.pc_op_mode_initial = 1                         # Operating
            if use_avg_flow:
                self.disp_pc_q0 = cd['Avg Q into cycle [MW]'][0]      # Not really, but close enough
            else:
                self.disp_pc_q0 = cd['Q into cycle [MW]'][0]      # Not really, but close enough
        else:
            self.pc_op_mode_initial = 3                 # Not Operating (may need to add startup later)
            
        # Power cycle targets
        max_pc_qin = (des_params.P_ref/des_params.design_eff)*des_params.cycle_max_frac
        self.q_pc_target_su_in = [max_pc_qin]*len(cd[list(cd.keys())[0]])   # TODO: adjust depending on results
        if use_avg_flow:
            self.q_pc_target_on_in = cd['Avg Q into cycle [MW]']
        else:
            self.q_pc_target_on_in = cd['Q into cycle [MW]']
        self.q_pc_max_in = [max_pc_qin]*len(cd[list(cd.keys())[0]])     

        ## Building receiver operation binary list
        ## using tank tank mass derivative to control receiver
        Nfb = 10     # Number of periods forward and backward for central difference (to remove noise in data)
        tstep = 24/len(cd['Hot Tank Mass [kg]'])
        dev = [0.]*Nfb
        for i in range(Nfb, len(cd['Hot Tank Mass [kg]']) - Nfb):
            dev.append((cd['Hot Tank Mass [kg]'][i+Nfb] - cd['Hot Tank Mass [kg]'][i-Nfb])/Nfb*tstep)
        dev.extend([0.]*Nfb) 

        rec_su = [0]
        tol = 250
        for d in dev[1:]:
            if d < -tol:            
                rec_su.append(0)    # Tank is draining
            elif d > tol:
                rec_su.append(1)    # Tank is filling
            else:
                if rec_su[-1] == 1:     # Tank filling has slowed but receiver is still on
                    rec_su.append(1)
                else:
                    rec_su.append(0)
        """
        # Old way of control
        prev_min = max(cd['Hot Tank Mass [kg]'])
        for i, mass in enumerate(cd['Hot Tank Mass [kg]'][240:]):
            if (mass - prev_min > 300000.) and cd['Cold Pumps [kW]'][i] > 100:  # Change in mass and pumps on
                break
            if mass < prev_min - 500.:
                prev_min = mass
                st_op = i
        if 'st_op' not in locals():
            st_op = cd['Hot Tank Temp [C]'].index(min(cd['Hot Tank Temp [C]']))     # minimum temp in hot tank marks start of receiver operation
        
        for i in range(len(cd['Cold Pumps [kW]'])):
            if i > st_op and cd['Cold Pumps [kW]'][i] > 100:
                rec_su.append(1)
            else:
                rec_su.append(0)
        """

        self.is_rec_su_allowed_in = rec_su

        ## Cycle start up binary
        if controller_adj:
            pc_su = np.array([1 if x > 25. else 0 for x in cd['Gross Power [MW]']])  # Is cycle running?
        else:
            pc_su = np.array([1 if x > 0 else 0 for x in cd['Gross Power [MW]']])  # Is cycle running?
        cycle_startup_time = self.startup_time*60
        for ind in np.where(pc_su[:-1] != pc_su[1:])[0]:        # cycle changes condition
            if pc_su[ind + 1] == 1:
                buffer = 20#10 # 20                                             #(10 minute buffer)
                pc_su[int(ind - cycle_startup_time + buffer): ind+1] = 1   # push start-up forward

        self.is_pc_su_allowed_in = pc_su.tolist()   # Is cycle running?
        self.is_pc_sb_allowed_in = [0]*len(cd[list(cd.keys())[0]])          # Cycle can not go into standby (using TES to keep cycle warm)
        self.is_elec_heat_dur_off = [1]*len(cd[list(cd.keys())[0]])         # Cycle always uses electric heaters in off mode

    def set_TES_temps(self, cd):
        self.T_tank_cold_init = cd['Cold Tank Temp [C]'][0]
        self.T_tank_hot_init = cd['Hot Tank Temp [C]'][0]
        
class simulation_parameters:
    def __init__(self, simulation_case = 1):
        self.use_transient_model = True  # Solve with transient receiver model?
        self.time_steps_per_hour = 60    # Simulation time resolution
        self.use_CD_attenuation = False   # Use CD visibility data and simple model to calculate attenuation
        self.use_mflow_per_path = False   # Use separate mass flow rates per flow circuit
        self.tracking_fraction_from_op_report = False   # If tracking fraction is included, use the value from the daily operations reports instead of the HFCS log files (typically lower tracking because some heliostats may be taken out of automatic control)
        self.override_receiver_Tout = False  # Over-ride computed receiver temperature solution and use Crescent Dunes outlet temperatures instead (should only be used with case 1)

        self.simulation_case = simulation_case
        if self.simulation_case == 1:  # Use Crescent Dunes tracking percentages, mass flow, inlet temperatures, and zero startup time so that the simulated receiver is on anytime that the CD receiver is on
        	self.include_tracking_fraction = True   # Use tracking fraction from CD data?
        	self.use_actual_mflow = True		    # Use receiver mass flow from CD data in place of internally calculated mflow?
        	self.use_actual_Tin = True				# Use receiver inlet T from CD data in place of cold tank temperature?
        	self.use_transient_startup = False		# Solve with transient receiver startup model?
        	self.use_zero_start_time = True    	    # Solve with no receiver start time (allows receiver to be "on" as soon as tracking fraction >0)
        	self.use_zero_piping_length = True       # Solve with near-zero riser and downcomer length? (CD data for inlet T is at receiver inlet. Use zero piping length to see timing of transients, but note that excluding piping will give an inaccurate pressure drop)

        elif self.simulation_case == 2:    # Use Crescent Dunes tracking percentages, inlet temperatures, and zero startup time, but allow ssc to set the mass flow rates
        	self.include_tracking_fraction = True   
        	self.use_actual_mflow =  False			 
        	self.use_actual_Tin = True			
        	self.use_transient_startup =  False	
        	self.use_zero_start_time = True    
        	self.use_zero_piping_length = False
                  
        elif self.simulation_case == 3:  # Allow ssc so set mass flow rates, inlet temperatures, and defocus levels.  Use average startup stage times from CD HFCS log files to set minimum startup stage times in ssc
        	self.include_tracking_fraction = False
        	self.use_actual_mflow =  False
        	self.use_actual_Tin = False
        	self.use_transient_startup = True 
        	self.use_zero_start_time =  False
        	self.use_zero_piping_length = False
        return

#============================================================================

# Get day of year
def get_day_of_year(date):
    year, month, day = [int(s) for s in date.split('_')]
    month -= 1
    day -= 1
    doy = day if month == 0 else np.cumsum([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])[month-1]+day
    return int(doy)


# Is this day within daylight savings time? 
def is_dst(date):
    year, month, day = [int(s) for s in date.split('_')]
    dstbounds = {2018:[69, 307], 2019:[68,306]}  # 2018 DST is 3/11 - 11/4, 2019 DST is 3/10 - 11/3
    doy = get_day_of_year(date)
    dst = False
    if year in dstbounds.keys() and  doy >= dstbounds[year][0] and doy <= dstbounds[year][1]:
        dst = True
    return dst

def check_resolution(nperhour):
    if nperhour not in [60, 60*4]:
        print ('Time resolution not recognized')
        return False
    return True



def get_start_point(date, nperhour):
    doy = get_day_of_year(date)
    dst = is_dst(date)
    h = doy*24
    if dst:
        h -=1    # CD data is in local time, shift simulation start hour back by one hour if this is during daylight savings time
    p = int(h*nperhour)
    return p    

def get_daily_data(date, nperhour, filename):
    p = get_start_point(date, nperhour)
    data = np.genfromtxt(filename, skip_header = p, max_rows = nperhour*24)
    return data


def get_weather_from_file(date, nperhour):
    year, month, day = [int(date.split('_')[i]) for i in range(3)]
    if not check_resolution(nperhour):
        return
    label = '1min' if nperhour == 60 else '15s' 
    filename = ('input_files/weather_files/ssc_weatherfile_%s_%d.csv'%(label,year))
    p = get_start_point(date, nperhour)
    data = np.genfromtxt(filename, delimiter = ',', skip_header = 3+p, max_rows = nperhour*24)

    weather = {}
    weather['dni'] = data[:,6]
    weather['tdry'] = data[:,8]
    weather['tdew'] = data[:,9]
    weather['wspd'] = data[:,12]
    
    return weather         


def get_clearsky(date, nperhour):
    year, month, day = [int(date.split('_')[i]) for i in range(3)]
    if not check_resolution(nperhour):
        return
    label = '1min' if nperhour == 60 else '15s'    
    filename = ('input_files/weather_files/clearsky_pvlib_ineichen_%s_%d.csv'%(label,year))
    return get_daily_data(date, nperhour, filename)    
    
def get_tracking(date, nperhour, is_from_op_report = False):
    year, month, day = [int(date.split('_')[i]) for i in range(3)]
    if not check_resolution(nperhour):
        return
    label = '1min' if nperhour == 60 else '15s'
    filename = ('input_files/tracking_%d_%s.csv'%(year, label))
    if is_from_op_report and year == 2018:
        filename = ('input_files/tracking_from_op_report_%d_%s.csv'%(year, label))    
    return get_daily_data(date, nperhour, filename)
    
def get_mflow(date, nperhour):
    year, month, day = [int(date.split('_')[i]) for i in range(3)]
    if not check_resolution(nperhour):
        return
    label = '1min' if nperhour == 60 else '15s'
    filename = ('input_files/mflow_%d_%s.csv'%(year,label))
    return get_daily_data(date, nperhour, filename)
    
def get_Tin(date, nperhour):
    year, month, day = [int(date.split('_')[i]) for i in range(3)]
    if not check_resolution(nperhour):
        return
    label = '1min' if nperhour == 60 else '15s'
    filename = ('input_files/Tin_%d_%s.csv'%(year,label))
    return get_daily_data(date, nperhour, filename)    


# Get daily reflectivity measurements
def get_reflectivity(date):
    year, month, day = [int(s) for s in date.split('_')]
    doy = get_day_of_year(date)
    refldata = np.genfromtxt('input_files/daily_refl_'+str(year)+'.csv', delimiter = ',', skip_header = 1)
    clean, soiled, clean_stdev, soiled_stdev = refldata[doy][1:5]/100
    return float(soiled), float(soiled_stdev)


# Get approximate daily heliostats offline
def get_heliostats_offline(date):
    doy = get_day_of_year(date)
    year, month, day = [int(s) for s in date.split('_')]
    offline = np.genfromtxt('input_files/daily_helios_offline_'+str(year)+'.csv')
    return offline[doy]
    

def get_visibility(date):
    doy = get_day_of_year(date)
    year, month, day = [int(s) for s in date.split('_')]
    offline = np.genfromtxt('input_files/daily_avg_visibility_'+str(year)+'.csv', skip_header = 1)
    return offline[doy]


def get_attenuation_coeffs(visibility):
    dist = np.linspace(0.25, 3.5, 14)  
    loss = 1.0 - 0.05**(dist/visibility)   # Model used at crescent dunes for attenuation loss
    x = np.ones((14,4))
    for j in range(1,4):
        x[:,j] = dist**j
    coeffs = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.matrix.transpose(x), x)), np.matrix.transpose(x)), loss) 
    return coeffs




#============================================================================

def get_filenames(year,sim_params):
    
    if sim_params.time_steps_per_hour == 60:
        label = '1min'
    elif sim_params.time_steps_per_hour == 60*4:
        label = '15s'
    else:
        print ('Error: Time resolution not recognized')
        return       
    
    weather = ('input_files/weather_files/ssc_weatherfile_%s_%d.csv'%(label, year))
    clearsky = ('input_files/weather_files/clearsky_pvlib_ineichen_%s_%d.csv'%(label, year))
    tracking = ('input_files/tracking_%d_%s.csv'%(year, label))
    mflow = ('input_files/mflow_%d_%s.csv'%(year,label))
    mflow_path1 = ('input_files/mflow_path1_%d_%s.csv'%(year,label))
    mflow_path2 = ('input_files/mflow_path2_%d_%s.csv'%(year,label))
    Tin = ('input_files/Tin_%d_%s.csv'%(year,label))   
    if sim_params.simulation_case == 3:
        Tin = ('input_files/Tin_%d_%s_290_when_notrack.csv'%(year,label))    
    
    Tout = ('input_files/Tout_%d_%s.csv'%(year,label))  
    
    return weather, clearsky, tracking, mflow, mflow_path1, mflow_path2, Tin, Tout



# Set ssc simulation on specifed date
def run_daily_simulation(date, design_params, properties, sim_params, init_TES = 30.0, add_vars = [], flux_maps = None, include_back_wall_T = False, output_file = None):
    year, month, day = [int(s) for s in date.split('_')]
    doy = get_day_of_year(date)
    dst = is_dst(date)
    starthr = doy*24 if not dst else doy*24 -1  # CD data is in local time, shift simulation start hour back by one hour if this is during daylight savings time

    heliostat_layout = np.genfromtxt('input_files/Crescent_Dunes_heliostat_layout.csv', delimiter = ',')
    nhel = heliostat_layout.shape[0]
    helio_positions = [heliostat_layout[j,0:2].tolist() for j in range(nhel)]

    weather_file, clearsky_file, tracking_file, mflow_file, mflow_path1_file, mflow_path2_file, Tin_file, Tout_file = get_filenames(year, sim_params)

    refl, reflstdev = get_reflectivity(date)
    if isinstance(properties.reflectivity, float):
        refl = properties.reflectivity
    elif properties.reflectivity == 'measured':
        refl = refl
    elif properties.reflectivity == 'plus_2sigma':
        refl += 2*reflstdev
    elif properties.reflectivity == 'minus_2sigma':
        refl -= 2*reflstdev
    else:
        print ('Error: Input parameter for reflectivity not recognized')
        return
    
       
    D0 = vars(design_params)   
    P0 = vars(properties)
    
    D = D0.copy()
    P = P0.copy()
    D.update(P)

    #initial TES level
    D['csp.pt.tes.init_hot_htf_percent'] = init_TES

    D['solar_resource_file'] = weather_file
    D['helio_positions'] = helio_positions
    D['solarm'] = D['Qrec'] * D['design_eff'] / D['P_ref']
    
    if flux_maps is None:
        D['field_model_type'] = 2
    else:
        D['field_model_type'] = 3
        D['N_hel'] = nhel
        D['eta_map_aod_format'] = False
        D['A_sf_in'] = flux_maps['A_sf']
        D['eta_map'] = flux_maps['eta_map']
        D['flux_maps'] = flux_maps['flux_maps']
        

    D['time_steps_per_hour'] = sim_params.time_steps_per_hour
    D['time_start'] = starthr*3600
    D['time_stop'] = (starthr+24)*3600   

    D['rec_clearsky_fraction'] = 1.0
    D['rec_clearsky_model'] = 0
    csky = np.genfromtxt(clearsky_file) 
    D['rec_clearsky_dni'] = csky.tolist()

    D['is_rec_model_trans'] = sim_params.use_transient_model
    D['is_rec_startup_trans'] = sim_params.use_transient_startup
    if sim_params.use_zero_start_time:
        D['is_rec_startup_trans'] = False
        D['rec_su_delay'] = 0.01
        D['rec_qf_delay'] = 0.01

    if sim_params.use_zero_piping_length:
        D['piping_length_mult'] = 0.0001

    nph = sim_params.time_steps_per_hour
    startpt = starthr*nph 
    
    if sim_params.use_actual_mflow:
        D['is_rec_user_mflow'] = True
        if sim_params.use_mflow_per_path:
            mflow1 = np.genfromtxt(mflow_path1_file)
            mflow2 = np.genfromtxt(mflow_path2_file)
            D['rec_user_mflow_path_1'] = mflow2.tolist()  # Note ssc path numbers are reversed relative to CD path numbers
            D['rec_user_mflow_path_2'] = mflow1.tolist()
            D['rec_control_per_path'] = True           
        else:
            mflow = np.genfromtxt(mflow_file)
            D['rec_user_mflow'] = mflow.tolist()  
            D['rec_control_per_path'] = False

    if sim_params.use_actual_Tin:
        D['is_rec_user_Tin'] = True
        Tin = np.genfromtxt(Tin_file)
        D['rec_user_Tin'] = Tin.tolist()          
    
    adjust = np.zeros(24*nph*365)
    fract_avail = 1.0 - get_heliostats_offline(date) / nhel   # Fraction of heliostats available for operation
    fract_refl = refl / D['helio_reflectance'] # Soiled reflectivity / clean reflectivity
    avail = fract_avail * fract_refl   
    avail_array = avail * np.ones(nph*24)   
    if sim_params.include_tracking_fraction:
        tracking_fraction = get_tracking(date, nph, sim_params.tracking_fraction_from_op_report)/100.    # note, updated tracking fraction files include small nonzero value between startup and postheat or drain command
        avail_array = tracking_fraction * fract_refl
        if not sim_params.tracking_fraction_from_op_report:  # Adjust tracking fraction for unavailable heliostats (note this is already included if tracking fraction is taken from operations report)
            avail_array *= fract_avail
        
        #i1 = np.where(tracking_fraction > 0)[0][0]
        #i2 = np.where(tracking_fraction > 0)[0][-1]
        #for j in range(nph*24):
        #    min_track = 0.001 if (j>i1 and j<i2) else 0.0    # Set very slight nonzero tracking fraction during some hours (avoids receiver shut-down during mid-day drops in tracking)
        #    avail_array[j] = np.maximum(tracking_fraction[j] * avail, min_track)   # Use a slight nonzero tracking fraction to allow receiver operation to continue if mass flow is specified, even when tracking drops to zero    
    adjust[startpt:startpt+24*nph] = 100 - (avail_array*100)
    D['sf_adjust:hourly'] = adjust.tolist()     

    if sim_params.include_tracking_fraction and 'is_rec_su_allowed_in' in D.keys():
        for i, op  in enumerate(D['is_rec_su_allowed_in']):
            if op == 1 and tracking_fraction[i] < 0.1002: #< 0.002:
                D['is_rec_su_allowed_in'][i] = 0  # shut off receiver
                if date == '2018_10_18' and i == 1058:
                    time_delay = 17 # minutes
                    D['is_rec_su_allowed_in'][i-time_delay:i] = [0]*time_delay    # TODO: custom to fix controller issue on this day

    
    if sim_params.use_CD_attenuation:
        vis = get_visibility(date)
        coeffs = get_attenuation_coeffs(vis)
        D['c_atm_0'] = float(coeffs[0])
        D['c_atm_1'] = float(coeffs[1])
        D['c_atm_2'] = float(coeffs[2])
        D['c_atm_3'] = float(coeffs[3])
        
    
    # Over-ride computed receiver temperature solution and use Crescent Dunes outlet temperatures instead
    if sim_params.override_receiver_Tout and sim_params.use_actual_mflow:  
        D['is_rec_user_Tout'] = True
        Tout = np.genfromtxt(Tout_file)
        D['rec_user_Tout'] = Tout.tolist()


    use_vars = ['beam', 'tdry', 'twet', 'wspd', 'q_sf_inc', 'eta_field', 'defocus', 'sf_adjust_out', 'q_dot_rec_inc', 
				'eta_therm', 'Q_thermal', 'm_dot_rec', 'q_startup', 'T_rec_in', 'T_rec_out', 'T_panel_out', 'q_piping_losses', 'q_thermal_loss', 
				'T_rec_out_end', 'T_panel_out_end', 'T_rec_out_max', 'T_panel_out_max', 'T_wall_rec_inlet', 'T_wall_rec_outlet', 'T_wall_riser', 'T_wall_downcomer', 'clearsky',
                'm_dot_rec_path_1', 'm_dot_rec_path_2', 'T_panel_out_path_1', 'T_panel_out_path_2',
                'T_tes_hot', 'T_tes_cold', 'mass_tes_cold', 'mass_tes_hot', 'q_dc_tes', 'q_ch_tes', 'e_ch_tes', 'tank_losses', 
				'm_dot_cr_to_tes_hot', 'm_dot_tes_hot_out', 'm_dot_pc_to_tes_cold', 'm_dot_tes_cold_out', 'm_dot_field_to_cycle', 'm_dot_cycle_to_field',
				'P_cycle','eta', 'T_pc_in', 'T_pc_out', 'q_pb', 'P_out_net', 'P_cycle_off_heat', 'P_tower_pump', 'htf_pump_power', 'P_cooling_tower_tot', 'P_fixed', 'P_plant_balance_tot', 'P_rec_heattrace', 'q_heater']
    
    if include_back_wall_T:
        D['T_report_location'] = 1
        use_vars = use_vars + ['T_htf_panel_%d'%j for j in range(min(D['N_panels'], 14))]
        

    for v in add_vars:
        if v not in use_vars:
            use_vars.append(v)

    R = call_ssc(D, use_vars, single_day = True)  

    Tavg = 0.5*( R['T_panel_out'] + R['T_rec_in'])
    cp = salt_cp(Tavg+273.15)
    R['Q_thermal_rec'] = R['m_dot_rec'] * cp * (R['T_panel_out'] -  R['T_rec_in'])/1.e6  # Qthermal at receiver exit (note Q_thermal from ssc is at downcomer exit)
    
    if 'm_dot_rec_path_1' not in R.keys() or len(R['m_dot_rec_path_1']) == 0:  # ssc data doesn't include per-path flow or outlet T
        R['m_dot_rec_path_1'] = np.zeros_like(R['m_dot_rec'])
        R['m_dot_rec_path_2'] = np.zeros_like(R['m_dot_rec'])
        R['T_panel_out_path_1'] = np.zeros_like(R['T_panel_out'])
        R['T_panel_out_path_2'] = np.zeros_like(R['T_panel_out'])
    
    if output_file is not None:
        savedata = np.array([R[k] for k in R.keys()]).transpose()
        header = ','.join(list(R.keys()))
        np.savetxt(output_file, savedata, delimiter = ',', fmt = '%.4f', header = header, comments = '')
    
    return R
    



def run_multiple_daily_simulations(dates, design_params, properties, sim_params, output_file = None):
    
    if sim_params.use_CD_attenuation:
        print ('Warning: CD attenuation model and visibility data cannot vary between days. Reverting to default ssc attenuation model')

    # Run single-day simulation for first day including simulation of flux maps
    print (dates[0])
    start = timeit.default_timer()
    R0 = run_daily_simulation(dates[0], design_params, properties, sim_params, add_vars = ['eta_map_out', 'flux_maps_for_import', 'A_sf'])
    flux = [x[2:] for x in R0['flux_maps_for_import']]
    flux_maps = {'A_sf':R0['A_sf'], 'eta_map': R0['eta_map_out'], 'flux_maps': flux}
    R0.pop('A_sf')
    R0.pop('eta_map_out')
    R0.pop('flux_maps_for_import')
    print ('Time = %.1fs'%(timeit.default_timer() - start))
    
    # Run subsequent simulations using flux maps computed during first simulation
    R = [R0]
    for d in range(1, len(dates)):
        print (dates[d])
        start = timeit.default_timer()
        Rnew = run_daily_simulation(dates[d], design_params, properties, sim_params, flux_maps = flux_maps)
        print ('Time = %.1fs'%(timeit.default_timer() - start))
        R.append(Rnew)
    return R        
        
    
    
    

# Set ssc input parameters from data in Dict
def set_ssc_data_from_dict(ssc_api, ssc_data, Dict):
    for key in Dict.keys():        
        try: 
            if type(Dict[key]) in [type(1),type(1.)]:
                ssc_api.data_set_number(ssc_data, key.encode("utf-8"), Dict[key])
            elif type(Dict[key]) == type(True):
                ssc_api.data_set_number(ssc_data, key.encode("utf-8"), 1 if Dict[key] else 0)
            elif type(Dict[key]) == type(""):
                ssc_api.data_set_string(ssc_data, key.encode("utf-8"), Dict[key].encode("utf-8"))
            elif type(Dict[key]) == type([]):
                if len(Dict[key]) > 0:
                    if type(Dict[key][0]) == type([]):
                        ssc_api.data_set_matrix(ssc_data, key.encode("utf-8"), Dict[key])
                    else:
                        ssc_api.data_set_array(ssc_data, key.encode("utf-8"), Dict[key])
                else:
                    #print ("Did not assign empty array " + key)
                    pass
            else:
               print ("Could not assign variable " + key )
               raise KeyError
        except:
            print ("Error assigning variable " + key + ": bad data type")


# Run ssc simulation using user-defined input parameters in dictionary D (all other input parameters will be set to default values in V)
def call_ssc(D, retvars = ['gen'], single_day = False):
    
    ssc = api.PySSC()
    dat = ssc.data_create()
 
    Vt = copy.deepcopy(V)
    Vt.update(D)
    
    set_ssc_data_from_dict(ssc, dat, Vt)

    mspt = ssc.module_create("tcsmolten_salt".encode("utf-8"))	
    if ssc.module_exec(mspt, dat) == 0:
        print ('Simulation error')
        idx = 1
        msg = ssc.module_log(mspt, 0)
        while (msg != None):
            print ('	: ' + msg.decode("utf-8"))
            msg = ssc.module_log(mspt, idx)
            idx = idx + 1

    R = {}
    for k in retvars:
        R[k] = ssc.data_get_array(dat, k.encode('utf-8'))
        
        if k in ['eta_map_out', 'flux_maps_out', 'flux_maps_for_import']:
            R[k] = ssc.data_get_matrix(dat, k.encode('utf-8'))
        elif k in ['A_sf']:
            R[k] = ssc.data_get_number(dat, k.encode('utf-8'))
        else:
            R[k] = ssc.data_get_array(dat, k.encode('utf-8'))
            if single_day:
                nph = int(len(R[k])/8760)
                R[k] = np.array(R[k][0:24*nph]) 

    ssc.module_free(mspt)
    ssc.data_free(dat)   
    return R
  







#=============================================================================
# Salt heat capacity and density
def salt_cp(T):  # T in K
    return 1000*(1.4387 + 5e-6*T + 2e-7*T*T - 1e-10*T*T*T)  

def salt_density(T): # T in K
    return 2299.4-0.7875*(T) + 2.e-4*(T**2) - 1.e-7*(T**3)



# Read in Crescent Dunes receiver data     
def read_CD_receiver_data(date, CD_data_direc = 'CD_receiver_data_files/'):

    try:
        data = np.genfromtxt(CD_data_direc+date + '_1s.csv', dtype = float, delimiter = ',', skip_header = 3)
        is_format_1 = True
    except:
        data = np.genfromtxt(CD_data_direc+date + '.csv', dtype = float, delimiter = ',', skip_header = 3)
        is_format_1 = False
        
    D = {}
    D['time'] = data[:,0] 

    if is_format_1:  # Data files with 1s resolution
        D['Tin1'], D['Tin2'], D['Tout1'], D['Tout2'], D['T_downc'], D['T_salt_pump_discharge'] = [(data[:,i]-32.)/1.8 for i in [7,8,9,10,13,2]]  # Convert temperatures from F to C
        D['Pin1'], D['Pin2'], D['P_downc_upper'], D['P_downc_lower'] = [data[:,i]*6894.76/1e6  for i in [5,6,11,12]]   # Convert pressures from psig to MPa
        D['mflow_clearsky'], D['mflow_RA'] = [data[:,i]*0.12599788 for i in [3,4]]    # Convert mass flow rates from kpph to kg/s
        
        dens = salt_density(D['T_salt_pump_discharge']+273.15)  # Density at salt discharge conditions
        D['mflow_pump_discharge'] =   (data[:,1] / 15852.) * dens # kg/s
        
        D['Q_HFCS'] = data[:,14]   # MW
        
        Tavg1 = 0.5*(D['Tout1']+D['Tin1'])+273.15  # Average T in circuit 1
        cavg1 = salt_cp(Tavg1)
        Tavg2 = 0.5*(D['Tout2']+D['Tin2'])+273.15   # Average T in circuit 2
        cavg2 = salt_cp(Tavg2)
        D['Q_mcpdT'] =  (0.5*D['mflow_RA']*cavg1*(D['Tout1']-D['Tin1']) + 0.5*D['mflow_RA']*cavg2*(D['Tout2']-D['Tin2']) )*1.e-6  # Expected thermal power in MW (from mcpdT)
        
        D['Tout'] = 0.5*(D['Tout1']+D['Tout2'])  # No data on mass flow per circuit, so just use average
        
        D['mflow_1'] = np.zeros_like(D['Tout'])
        D['mflow_2'] = np.zeros_like(D['Tout'])
        
    else:
        D['Tin1'], D['Tin2'] = [(data[:,i]-32.)/1.8 for i in [4,5]]  # Convert temperatures from F to C
        D['Tout1a'], D['Tout1b'], D['Tout1c'] =  [(data[:,i]-32.)/1.8 for i in [6,7,8]] 
        D['Tout2a'], D['Tout2b'], D['Tout2c'] =  [(data[:,i]-32.)/1.8 for i in [9,10,11]] 
        D['T_downc_a'], D['T_downc_b'], D['T_downc_c'], D['T_downc_d']=  [(data[:,i]-32.)/1.8 for i in [12, 13, 14, 15]] 
        
        D['Tout1'] = D['Tout1a']
        D['Tout2'] = D['Tout2a']
        D['T_downc'] = D['T_downc_a']
  
        D['Q_HFCS'] = data[:,16]   # MW
        D['mflow_RA'] = data[:,1]*0.12599788   # Convert mass flow rates from kpph to kg/s
        D['mflow_1'] = D['mflow_RA'] * (data[:,2]/np.maximum((data[:,2]+data[:,3]), 1.e-6))
        D['mflow_2'] = D['mflow_RA'] * (data[:,3]/np.maximum((data[:,2]+data[:,3]), 1.e-6))

        keys = ['helio_avail', 'helio_stow', 'helio_standby', 'helio_preheat', 'helio_track', 'helio_postheat', 'helio_offline', 'helio_transition']
        for j in range(8):
            D[keys[j]] = data[:,17+j]
            
        Tavg1 = 0.5*(D['Tout1']+D['Tin1'])+273.15  # Average T in circuit 1
        cavg1 = salt_cp(Tavg1)
        Tavg2 = 0.5*(D['Tout2']+D['Tin2'])+273.15   # Average T in circuit 2
        cavg2 = salt_cp(Tavg2)
        #D['Q_mcpdT'] =  (0.5*D['mflow_RA']*cavg1*(D['Tout1']-D['Tin1']) + 0.5*D['mflow_RA']*cavg2*(D['Tout2']-D['Tin2']) )*1.e-6  # Expected thermal power in MW (from mcpdT)
        D['Q_mcpdT'] =  (D['mflow_1']*cavg1*(D['Tout1']-D['Tin1']) + D['mflow_2']*cavg2*(D['Tout2']-D['Tin2']) )*1.e-6  # Expected thermal power in MW (from mcpdT)
        
        
        #--- Calculate mass-averaged exit temperature (where possible)
        m1 = np.nan_to_num(D['mflow_1'], copy = True, nan = 0.0 )
        m2 = np.nan_to_num(D['mflow_2'], copy = True, nan = 0.0 )
        
        D['Tout'] = np.zeros_like(D['Tout1'])
        ok = np.logical_and(m1>0, m2>0)
        inds = np.where(ok)[0]
        D['Tout'][inds] = (m1[inds]*D['Tout1'][inds] + m2[inds]*D['Tout2'][inds])/(m1[inds]+m2[inds])
        inds = np.where(ok == False)[0]
        D['Tout'][inds] = 0.5*(D['Tout1'][inds]+D['Tout2'][inds])  # No data on mass flow per circuit, so just use average

    
    # Update nan values to most recent previous value
    for k in D.keys():  
        inds = np.where(np.isnan(D[k]))[0]
        for j in inds:
            D[k][j] = D[k][j-1] if j>0 else 0.0
            
    return D

def convert2datetime(data):
    dt = datetime.datetime.fromordinal(datetime.datetime(1900, 1, 1).toordinal() + int(data) - 2)
    hour, minute, second = floatHourToTime(data % 1)
    dt = dt.replace(hour=hour, minute=minute, second=second)
    return dt

def floatHourToTime(fh):
    h, r = divmod(fh*24, 1)
    m, r = divmod(r*60, 1)
    return (
        int(h),
        int(m),
        int(r*60),
    )

# Read in CD cycle data from pickle or make pickle
def read_CD_cycle_data(des_params, CD_cycle_data_dir = "../sgs/CD_data/daily_reports/", pklname = "data_dict.pkl", inter_missing_data = True, use_avg_salt_dens = True, flow_avg_wind = 20, dates_w_controller_adj = []):
    ## Pulls all data from daily_reports file (a bit slow without pickle)

    if os.path.exists(CD_cycle_data_dir + pklname):
        data = pickle.load(open(CD_cycle_data_dir + pklname, "rb"))
    else:
        # Pulling data
        datafiles = os.listdir(CD_cycle_data_dir)
        basestr = 'CDS Daily Operations Report '
        ftype = '.xlsb'
        data = {}
        fcount = 0
        for f in datafiles:
            if basestr in f:
                dkey = f.replace(basestr,'')
                dkey = dkey.replace(ftype,'')
                dkey = dkey.replace(' ', '_')
                fpath = CD_cycle_data_dir + f
                print("Reading in Day " + dkey)
                print("{:.2f} % Complete".format(100*fcount/len(datafiles)))
                data[dkey] = pd.read_excel( fpath, sheet_name = 'Chart2', engine = 'pyxlsb', header = 3, skiprows=[4], nrows = 1440)#1440)
                ## Fix dataframe (shift headers on a subset of columns)
                bad_col_name = "Cold Salt Pmps Disch Flow"
                new_col_names = {}
                is_shift_cols = False
                ### DO NOT change column order between read_excel and this loop 
                for c in data[dkey].columns:
                    if c == bad_col_name:
                        is_shift_cols = True
                        prev_col = c
                    elif not is_shift_cols or c == 'TOTAL RA SYSTEM FLOW':  # Last column is correct
                        new_col_names[c] = c # data and column name are correct
                    else:
                        new_col_names[prev_col] = c     # Shift column over NOTE: pd.shift did not work for this...?
                        prev_col = c
                data[dkey] = data[dkey].drop(labels = ["Pond Drain Pump 2"], axis = 'columns')        # This column data is lost
                data[dkey] = data[dkey].rename(columns = new_col_names)
                data[dkey] = data[dkey].set_index('Date')
                fcount += 1

            pickle.dump(data, open(CD_cycle_data_dir + pklname, "wb") )

    # Creating output data structure and converting data
    D = {}
    maxtank_level = 0.0
    mintank_level = 10.0
    mintank_avg = []
    maxtank_avg = []
    for day in data.keys():
        # Filling missing data 
        if inter_missing_data:
            data[day] = data[day].interpolate(method='quadratic')

        # Fixing keys to be 'year_month_day' format (for older pkl)
        if '_' not in day:
            newkey = day.replace(' ', '_')
        else:
            newkey = day

        D[newkey] = pd.DataFrame()
        D[newkey]['Gross Power [MW]'] = data[day]['Gen Gross Active Power']
        D[newkey]['Net Power [MW]'] = data[day]['Unit Net MW']
        D[newkey]['Aux Power (Calc) [kW]'] = data[day]['Aux Transf 1 Actv Pwr'] + data[day]['Aux Trnsfmr 2 Actv Pwr']
        D[newkey]['Cold Pumps [kW]'] = data[day]['Cold Salt Pump 1'] + data[day]['Cold Salt Pump 2'] +  data[day]['Cold Salt Pump 3'] + data[day]['Cold Salt Pump 4']

        D[newkey]['Hot Tank Temp [C]'] = (data[day]['Hot Salt Tank Standpipe Av Tmp'] - 32.)*5./9.         # F -> C
        D[newkey]['Cold Tank Temp [C]'] = (data[day]['Cold Salt Tank Standpipe Av Tmp'] - 32.)*5./9.        # F -> C
        D[newkey]['Hot Tank Level [m]'] = data[day]['Hot Salt Tank Level Seltd']/3.28084                    # ft -> m
        D[newkey]['Cold Tank Level [m]'] = data[day]['Col Salt Tank Level Seltd']/3.28084                   # ft -> m

        D[newkey]['Ambient Temp [F]'] = 0.25*(data[day]['North Field Temperature'] + data[day]['East Field Temperature'] 
                                                + data[day]['South Field Temperature '] + data[day]['West Field Temperature'])
        D[newkey]['Ambient Temp [C]'] = (D[newkey]['Ambient Temp [F]'] - 32.)*5./9.         # F -> C

        ### Cycle return temperature - cleaning data
        salt_outs = ['Econ 1 Salt Out Tmp', 'Econ 2 Salt Out Tmp']
        for s_out_temp in salt_outs:
            data[day].loc[(data[day][s_out_temp]<0.001), s_out_temp] = np.NaN # replacing all 0 with NaN (bad data) (NOTE: temperature never is close to 0 in this data)
            data[day][s_out_temp].interpolate()

        ### Filtering data to remove the train out of operation from average calc.
        D[newkey]['Cycle Return Temp [F]'] = 0.5*(data[day]['Econ 1 Salt Out Tmp'] + data[day]['Econ 2 Salt Out Tmp'])
        mask = ((data[day]['Econ 1 Salt Out Tmp'] - data[day]['Econ 2 Salt Out Tmp']) > 5.)
        D[newkey].loc[mask, 'Cycle Return Temp [F]'] = data[day]['Econ 1 Salt Out Tmp']           # Note: this could be a little bit better
        mask = ((data[day]['Econ 2 Salt Out Tmp'] - data[day]['Econ 1 Salt Out Tmp']) > 5.)
        D[newkey].loc[mask, 'Cycle Return Temp [F]'] = data[day]['Econ 2 Salt Out Tmp']
        D[newkey]['Cycle Return Temp [C]'] = (D[newkey]['Cycle Return Temp [F]'] - 32.)*5./9.        # F -> C

        ### CD tank conditions -> salt mass in tank
        tank_radius = 21.336    # [m]
        Tavg = 0.5*(D[newkey]['Hot Tank Temp [C]'] + D[newkey]['Cold Tank Temp [C]']) + 273.15
        if use_avg_salt_dens:
            dens_h = salt_density(Tavg)
            dens_c = dens_h
        else:
            dens_h = salt_density(D[newkey]['Hot Tank Temp [C]']+273.15)  # Density in hot tank
            dens_c = salt_density(D[newkey]['Cold Tank Temp [C]']+273.15)  # Density in cold tank

        D[newkey]['Hot Tank Mass [kg]'] = dens_h*D[newkey]['Hot Tank Level [m]']*np.pi*(tank_radius**2)
        D[newkey]['Cold Tank Mass [kg]'] = dens_c*D[newkey]['Cold Tank Level [m]']*np.pi*(tank_radius**2)

        ### Heat into cycle
        D[newkey]['Flow into cycle [kg/s]'] = (data[day]['Superheater 1 Flow'] + data[day]['Reheater 1 Flow'] 
                                                + data[day]['Superheater 2 Flow'] + data[day]['Reheater 2 Flow'])/2.20462       # lbs/s -> kg/s
        ### Determine if trains are operational
        pwr =     data[day]['Gen Gross Active Power']/max(data[day]['Gen Gross Active Power'])
        t1_flow = (data[day]['Superheater 1 Flow'] + data[day]['Reheater 1 Flow'])/max(D[newkey]['Flow into cycle [kg/s]']) # avg 
        t2_flow = (data[day]['Superheater 2 Flow'] + data[day]['Reheater 2 Flow'])/max(D[newkey]['Flow into cycle [kg/s]'])

        if newkey in dates_w_controller_adj:
            D[newkey].loc[pwr <= 0.25, 'Flow into cycle [kg/s]'] = 0.0
            mask = (pwr > 0.25) & ((t1_flow - t2_flow) > 0.03)    # generating power, train 1 flow is 3% greater then train 2
        else:
            D[newkey].loc[pwr <= 0.01, 'Flow into cycle [kg/s]'] = 0.0
            mask = (pwr > 0.0) & ((t1_flow - t2_flow) > 0.03)    # generating power, train 1 flow is 3% greater then train 2

        D[newkey].loc[mask, 'Flow into cycle [kg/s]'] = (data[day]['Superheater 1 Flow'] + data[day]['Reheater 1 Flow']).loc[mask]/2.20462  
        if newkey in dates_w_controller_adj:
            mask = (pwr > 0.25) & ((t2_flow - t1_flow) > 0.03)    # generating power, train 2 flow is 3% greater then train 1
        else:
            mask = (pwr > 0.0) & ((t2_flow - t1_flow) > 0.03)    # generating power, train 2 flow is 3% greater then train 1
        D[newkey].loc[mask, 'Flow into cycle [kg/s]'] = (data[day]['Superheater 2 Flow'] + data[day]['Reheater 2 Flow']).loc[mask]/2.20462  
        ### If pump power is minimal, flow is zero
        #D[newkey].loc[(data[day]['Hot Salt Pump 1'] + data[day]['Hot Salt Pump 2'] + data[day]['Hot Salt Pump 3'])<22.0, 'Flow into cycle [kg/s]'] = 0.0

        #### Rolling avgerage to set flow rate
        df = D[newkey]['Flow into cycle [kg/s]'].expanding().mean()
        df2 = D[newkey]['Flow into cycle [kg/s]'].rolling(window=flow_avg_wind).mean()  #center=True  # TODO: average over center
        #D[newkey]['Avg flow into cycle [kg/s]'] = pd.concat([df.iloc[0:flow_avg_wind], df2.iloc[flow_avg_wind:]])
        D[newkey]['Avg flow into cycle [kg/s]'] = pd.concat([df.iloc[0:flow_avg_wind], df2.iloc[flow_avg_wind:]])

        Tavg = 0.5*(D[newkey]['Hot Tank Temp [C]'] + D[newkey]['Cycle Return Temp [C]']) + 273.15 # update average temp
        cavg = salt_cp(Tavg)
        
        D[newkey]['Q into cycle [MW]'] = D[newkey]['Flow into cycle [kg/s]']*cavg*(D[newkey]['Hot Tank Temp [C]'] - D[newkey]['Cycle Return Temp [C]'])*1.e-6
        D[newkey]['Avg Q into cycle [MW]'] = D[newkey]['Avg flow into cycle [kg/s]']*cavg*(D[newkey]['Hot Tank Temp [C]'] - D[newkey]['Cycle Return Temp [C]'])*1.e-6   
        
        #D[newkey]['Q into cycle [MW]'] = D[newkey]['Flow into cycle [kg/s]']*cavg*(D[newkey]['Hot Tank Temp [C]'] - 290.)*1.e-6
        #D[newkey]['Avg Q into cycle [MW]'] = D[newkey]['Avg flow into cycle [kg/s]']*cavg*(D[newkey]['Hot Tank Temp [C]'] - 290)*1.e-6   

        # TODO: filtering out low heat input to fix controller issues
        min_heat = (des_params.P_ref/des_params.design_eff)*0.2


        D[newkey]['E charge TES [MWht]'] = D[newkey]['Hot Tank Mass [kg]']*cavg*(D[newkey]['Hot Tank Temp [C]'] - des_params.T_htf_cold_des)*1.e-6/(3600.)    # J -> MJ -> MWh

        ### Finding max and min tank levels:
        maxtank_level = max(maxtank_level, max(D[newkey]['Hot Tank Level [m]']), max(D[newkey]['Cold Tank Level [m]']))
        mintank_level = min(mintank_level, min(D[newkey]['Hot Tank Level [m]']), min(D[newkey]['Cold Tank Level [m]']))

        mintank_avg.append(min( min(D[newkey]['Hot Tank Level [m]']), min(D[newkey]['Cold Tank Level [m]'])))
        maxtank_avg.append(max( max(D[newkey]['Hot Tank Level [m]']), max(D[newkey]['Cold Tank Level [m]'])))

        D[newkey] = D[newkey].to_dict('list')

    D['min_max_tank_levels'] = [mintank_level, maxtank_level]
    D['avg_min_max_tank_levels'] = [sum(mintank_avg)/len(mintank_avg), sum(maxtank_avg)/len(maxtank_avg)]


    return D

def estimate_init_TES(CD, tank_bounds):
    init_tes = 100*(CD['Hot Tank Level [m]'][0] - tank_bounds[0])/(tank_bounds[1] - tank_bounds[0])
    #init_tes = max(init_tes - 1.0, 0.0) # adjustment factor
    #init_tes = max(init_tes - 2.5, 0.0) # adjustment factor (older)
    return init_tes

# Read stored ssc data
def read_ssc_data(date, direc):
    ssc_file = direc + 'ssc_' + date + '.csv'
    data = np.genfromtxt(ssc_file, delimiter = ',', skip_header = 1)
    cols = np.genfromtxt(ssc_file, delimiter = ',', max_rows = 1, dtype = 'str')
    ncols = len(cols)
    sscdata = {cols[j]:data[:,j] for j in range(ncols)}
    if 'Q_thermal_rec' not in sscdata.keys():
        Tavg = 0.5*( sscdata['T_panel_out'] +  sscdata['T_rec_in'])
        cp = salt_cp(Tavg+273.15)
        sscdata['Q_thermal_rec'] = sscdata['m_dot_rec'] * cp * (sscdata['T_panel_out'] -  sscdata['T_rec_in'])/1.e6  
    return sscdata


# Calculate average value in CD data at each ssc timestep
def calc_CD_avg(CD, sscstep):
    CDstep = np.diff(CD['time']).mean()   # CD data timestep
    nperday = int(24/sscstep)

    keys = ['mflow_RA', 'mflow_1', 'mflow_2', 'Q_mcpdT', 'Tout', 'Tout1', 'Tout2']
    new_names = ['mflow', 'mflow1', 'mflow2', 'Q_thermal', 'Tout', 'Tout1', 'Tout2']
    nvar = len(keys)

    # 1-min full-day time resolution in CD data
    if len(CD['mflow_RA']) == 1440 and np.abs(CDstep*60 - 1) < 0.01:   # 1-min full-day time resolution in CD data   
        if np.abs(sscstep*60 - 1) < 0.01:    # 1-min time resolution in ssc data
            CDavg = {new_names[i]:CD[keys[i]] for i in range(nvar)}
            if 'Gross Power [MW]' in CD:    #Add Cycle information
                CDavg['Gross Power [MW]'] = CD['Gross Power [MW]']
                CDavg['E charge TES [MWht]'] = CD['E charge TES [MWht]'] 
        else:  # 15-second time resolution in ssc data... use same CD value at all ssc time points until next time step
            CDavg = {new_names[i]:np.repeat(CD[keys[i]], 4) for i in range(nvar)}  
            
    # 1-second part-day time resolution in CD data
    else:
        CDavg = {k:np.zeros(nperday) for k in new_names} # CD data averaged over ssc timestep
        tol = 0.01*sscstep   # Time tolerance (hr)
        for j in range(nperday):
            hourstart = j * sscstep
            inds = np.where(np.logical_and(CD['time'] >= hourstart-tol, CD['time'] <= hourstart + sscstep)== True)[0]
            if len(inds)>0:
                for i in range(nvar):
                    CDavg[new_names[i]][j] = CD[keys[i]][inds].mean()
    return CDavg



# Calculate integrated generation, mass flow and mass-weighted average outlet temperature from CD data
def calc_CD_integrated(CD, date):  
    step = np.diff(CD['time']).mean()   # time step (hr) in CD data
    keys = ['mflow_RA', 'mflow_1', 'mflow_2', 'Q_mcpdT', 'Tout', 'Tout1', 'Tout2']
    new_names = ['mflow', 'mflow1', 'mflow2', 'Q', 'Tout', 'Tout1', 'Tout2']
    data = {new_names[i]: np.nan_to_num(CD[keys[i]], True, 0.0) for i in range(len(keys))}

    # Ignore time points without heliostats tracking for now
    nph_track = 60
    tracking = get_tracking(date, nph_track) # Read tracking data (only using points with nonzero tracking in summation)
    track_step = (CD['time'] * nph_track).astype(int)
    inds = np.where(tracking[track_step]<0.001)[0]   
    for k in new_names:
        data[k][inds] = 0.0


    if 'Gross Power [MW]' in CD: # power cycle and tes
        power = np.nan_to_num(CD['Gross Power [MW]'], True, 0.0)
        TES = np.nan_to_num(CD['E charge TES [MWht]'], True, 0.0)
        # Ignore time points when TES is empty < 300 MWht
        inds = np.where(TES < 300.0)[0]
        TES[inds] = 0.0
    
    CDtot = {}  
    CDtot['Q_thermal'] = np.cumsum(data['Q'])*step  # Integrated thermal power (MWht)
    for k in ['mflow', 'mflow1', 'mflow2']:
        CDtot[k] = np.cumsum(data[k])*step*3600   # Integrated mass flow through receiver (kg)
    
    CDtot['Tout'] = np.cumsum(data['mflow']*data['Tout'])*step*3600 / np.maximum(CDtot['mflow'], 1.e-6) # Mass-weighted average exit temperature (C)
    
    if data['mflow1'].max() > 0.0:  # CD data set contains mass flow rates per math -> Mass-weighted-average per-path outlet T based on per-path mass flow
        CDtot['Tout1'] = np.cumsum(data['mflow1']*data['Tout1'])*step*3600 / np.maximum(CDtot['mflow1'], 1.e-6) # Mass-weighted average exit temperature (C)
        CDtot['Tout2'] = np.cumsum(data['mflow2']*data['Tout2'])*step*3600 / np.maximum(CDtot['mflow2'], 1.e-6) # Mass-weighted average exit temperature (C)
    else:  # CD data set contains mass flow rates per math -> Mass-weighted-average per-path outlet T based on total mass flow
        CDtot['Tout1'] = np.cumsum(data['mflow']*data['Tout1'])*step*3600 / np.maximum(CDtot['mflow'], 1.e-6) # Mass-weighted average exit temperature (C)
        CDtot['Tout2'] = np.cumsum(data['mflow']*data['Tout2'])*step*3600 / np.maximum(CDtot['mflow'], 1.e-6) # Mass-weighted average exit temperature (C)

    if 'Gross Power [MW]' in CD:
        CDtot['Gross Gen [MWhe]'] = np.cumsum(power)*step  # Integrated cycle power output (MWhe)
        CDtot['E charge TES [MWht]'] = TES    # TES charge state (MWht) for now

    return CDtot


# Calculate integrated generation, mass flow and mass-weighted average outlet temperature from ssc data
def calc_ssc_integrated(ssc, date):
    step = 24 / len(ssc['beam'])  # Time step in ssc data
    nperhour = int(len(ssc['beam'])/24)
    
    keys = ['m_dot_rec', 'm_dot_rec_path_1', 'm_dot_rec_path_2', 'Q_thermal', 'T_panel_out', 'T_panel_out_path_1', 'T_panel_out_path_2']
    new_names = ['mflow', 'mflow2', 'mflow1', 'Q_thermal', 'Tout', 'Tout2', 'Tout1']  # Note ssc flow paths are numbered opposite of Crescent Dunes flow paths
    data = {new_names[i]: np.copy(ssc[keys[i]]) for i in range(len(keys))}
 
    # Ignore time points without heliostats tracking and without receiver operation for now
    tracking = get_tracking(date, nperhour)
    inds = np.where(np.logical_or(tracking<0.001, np.abs(data['Q_thermal'])<0.001) == True)[0] 
    for k in new_names:
        data[k][inds] = 0.0

    # cycle and TES
    power = np.nan_to_num(ssc['P_cycle'], True, 0.0)
    TES = np.nan_to_num(ssc['e_ch_tes'], True, 0.0)
    # Ignore time points when TES is empty < 300 MWht
    inds = np.where(TES < 300.0)[0]
    TES[inds] = 0.0
    
    ssctot = {}  
    ssctot['Q_thermal'] = np.cumsum(data['Q_thermal'])*step        # Integrated thermal power (MWht)   
    for k in ['', '1', '2']:
        ssctot['mflow'+k] = np.cumsum(data['mflow'+k])*step*3600   # Integrated mass flow through receiver (kg)        
        ssctot['Tout'+k] = np.cumsum(data['mflow'+k]*data['Tout'+k])*step*3600 / np.maximum(ssctot['mflow'+k], 1.e-6) # Mass-weighted average exit temperature (C)
    ssctot['Gross Gen [MWhe]'] = np.cumsum(power)*step  # Integrated cycle power output (MWhe)
    ssctot['E charge TES [MWht]'] = TES                 # TES charge state (MWht) with empty conditions = 0.0 for now

    return ssctot

# Calculate errors 
def calc_errors(ssc, CD, date, des_params):
    nperhour = int(len(ssc['beam'])/24)
    sscstep =  1./nperhour
    
    CDavg = calc_CD_avg(CD, sscstep)
    CDint = calc_CD_integrated(CD, date)
    sscint = calc_ssc_integrated(ssc, date)

    # Calculate errors in time-integrated thermal power, mass-wtd-avg outlet temperature, total mass flow through receiver
    errors_int = {}
    errors_int['Q_thermal'] = (sscint['Q_thermal'][-1] - CDint['Q_thermal'][-1])/CDint['Q_thermal'][-1]
    errors_int['mflow'] = (sscint['mflow'][-1] - CDint['mflow'][-1])/CDint['mflow'][-1]
    errors_int['Tout'] = (sscint['Tout'][-1] - CDint['Tout'][-1])    

    if CDavg['mflow1'].max() > 0.0:  # Flow path mass flow data is available
        errors_int['mflow1'] = (sscint['mflow1'][-1] - CDint['mflow1'][-1])/CDint['mflow1'][-1]
        errors_int['mflow2'] = (sscint['mflow2'][-1] - CDint['mflow2'][-1])/CDint['mflow2'][-1]
        errors_int['Tout1'] = (sscint['Tout1'][-1] - CDint['Tout1'][-1])
        errors_int['Tout2'] = (sscint['Tout2'][-1] - CDint['Tout2'][-1])
    else:
        errors_int['mflow1'] = errors_int['mflow']
        errors_int['mflow2'] = errors_int['mflow']
        errors_int['Tout1'] = errors_int['Tout']
        errors_int['Tout2'] = errors_int['Tout']   
                                       
    
    if 'Gross Power [MW]' in CD:
        # Determine TES capacity
        TES_cap = max(CD['Cold Tank Mass [kg]'])*salt_cp((des_params.T_htf_cold_des +  des_params.T_htf_hot_des)/2 + 273.15)*(des_params.T_htf_hot_des - des_params.T_htf_cold_des)*1.e-6/(3600.)   # J -> MJ -> MWh

        # Calculate errors in time-integrated gross cycle power and peak TES charge state
        errors_int['gen'] = (sscint['Gross Gen [MWhe]'][-1] - CDint['Gross Gen [MWhe]'][-1])/CDint['Gross Gen [MWhe]'][-1]
        errors_int['peak_tes'] = (max(sscint['E charge TES [MWht]']) - max(CDint['E charge TES [MWht]']))/TES_cap     # peak tes error/ TES capacity
        errors_int['avg_abserr_cap'] = (abs(sscint['E charge TES [MWht]'] - CDint['E charge TES [MWht]'])).mean()/TES_cap   # average tes error / TES capacity
        errors_int['avg_err_cap'] = (sscint['E charge TES [MWht]'] - CDint['E charge TES [MWht]']).mean()/TES_cap   # average tes error / TES capacity


    # Calculate RMS errors in thermal power, outlet temperature, mass flow rate (only using time points with ssc model receiver operation)
    errors_rms = {}
    nonzero = np.where(np.abs(ssc['Q_thermal'])>0.0)[0]  # ssc time points with receiver operation
    errors_rms['Q_thermal'] = (((ssc['Q_thermal'][nonzero] - CDavg['Q_thermal'][nonzero])**2).mean())**0.5 / CDavg['Q_thermal'][nonzero].mean()  # Time point RMS error in Q_thermal / average Q_thermal
    errors_rms['mflow'] = (((ssc['m_dot_rec'][nonzero] - CDavg['mflow'][nonzero])**2).mean())**0.5 / CDavg['mflow'][nonzero].mean()  # Time point RMS error in mass flow / average mass flow
    errors_rms['Tout'] = (((ssc['T_panel_out'][nonzero] - CDavg['Tout'][nonzero])**2).mean())**0.5  # Time point RMS error in outlet temperature

    if CDavg['mflow1'].max() > 0.0:  # Flow path mass flow data is available
        errors_rms['mflow1'] = (((ssc['m_dot_rec_path_2'][nonzero] - CDavg['mflow1'][nonzero])**2).mean())**0.5 / CDavg['mflow1'][nonzero].mean()  # Time point RMS error in mass flow / average mass flow
        errors_rms['mflow2'] = (((ssc['m_dot_rec_path_1'][nonzero] - CDavg['mflow2'][nonzero])**2).mean())**0.5 / CDavg['mflow2'][nonzero].mean()  
        errors_rms['Tout1'] = (((ssc['T_panel_out_path_2'][nonzero] - CDavg['Tout1'][nonzero])**2).mean())**0.5  # Time point RMS error in outlet temperature
        errors_rms['Tout2'] = (((ssc['T_panel_out_path_1'][nonzero] - CDavg['Tout2'][nonzero])**2).mean())**0.5 
    else:
        errors_rms['mflow1'] = errors_rms['mflow']
        errors_rms['mflow2'] = errors_rms['mflow']
        errors_rms['Tout1'] = errors_rms['Tout']
        errors_rms['Tout2'] = errors_rms['Tout']   


    if 'Gross Power [MW]' in CD:
        # Calculate RMS errors in gross power, TES charge state
        nonzero = np.where(ssc['P_cycle']>0.0)[0]   # ssc time points with cycle operation
        errors_rms['power'] = (((ssc['P_cycle'][nonzero]  - CDavg['Gross Power [MW]'][nonzero])**2).mean())**0.5 / CDavg['Gross Power [MW]'][nonzero].mean() # Time point RMS error in gross power / average gross power
        nonzero = np.where(ssc['e_ch_tes']>300.0)[0]  # ssc time points with TES charge
        errors_rms['TES'] = (((ssc['e_ch_tes'][nonzero] - CDavg['E charge TES [MWht]'][nonzero])**2).mean())**0.5 / CDavg['E charge TES [MWht]'][nonzero].mean()  # Time point RMS error in TES / average TES

    return errors_int, errors_rms

    

    

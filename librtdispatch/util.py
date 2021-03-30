import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
from math import ceil, cos
import pandas as pd
import pickle
import zipfile
import datetime
from pvlib import solarposition

try:
    import mediation.plant as plant
except:           # if called from case_study
    import loredash.mediation.plant as plant

def write_dict_as_csv(mydict, csv_path='./mydict.csv'):
    import csv
    with open(csv_path, 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in mydict.items():
            writer.writerow([key, value])

# Get day of year (0-indexed, date is datetime.datetime)
def get_day_of_year(date):
    return (date - datetime.datetime(date.year, 1, 1)).days

# Get number of seconds elapsed with since the beginning of the year
def get_time_of_year(date):
    return (date - datetime.datetime(date.year,1,1,0,0,0)).total_seconds()

# Get number of seconds elapsed with since the beginning of the day
def get_time_of_day(date):
    return (date- datetime.datetime(date.year,date.month,date.day,0,0,0)).total_seconds()
        

# Get datetime.datetime from day of year (0-indexed)
def get_date(doy, year):
    return datetime.datetime(year,1,1) + datetime.timedelta(days = doy)

# Is this day within daylight savings time? 
def is_dst(date):
    dstbounds = {2018:[69, 307], 2019:[68,306]}  # 2018 DST is 3/11 - 11/4, 2019 DST is 3/10 - 11/3
    doy = get_day_of_year(date)
    dst = False
    if date.year in dstbounds.keys() and  doy >= dstbounds[date.year][0] and doy <= dstbounds[date.year][1]:
        dst = True
    return dst

def get_clearsky_data(clearsky_file, time_steps_per_hour):
    clearsky_data = np.genfromtxt(os.path.join(os.path.dirname(__file__), clearsky_file))
    if time_steps_per_hour != 60:
        clearsky_data = np.array(translate_to_new_timestep(clearsky_data, 1./60, 1./time_steps_per_hour))
    return clearsky_data

def get_ground_truth_weather_data(ground_truth_weather_file, time_steps_per_hour):
    ground_truth_weather_data = read_weather_data(ground_truth_weather_file)
    if time_steps_per_hour != 60:
        ground_truth_weather_data = update_weather_timestep(ground_truth_weather_data, time_steps_per_hour)
    return ground_truth_weather_data

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
            adjust = get_field_adjustment_from_CD_data(year, plant_design['N_hel'], plant_design['helio_reflectance']*100, True, None, False)            
        else:
            adjust = (fixed_soiling_loss * 100 * np.ones(steps_per_hour*24*365))  

    elif control_field == 'CD_data':
        if use_CD_measured_reflectivity:
            adjust = get_field_adjustment_from_CD_data(year, plant_design['N_hel'], plant_design['helio_reflectance']*100, True, None, True)
        else:
            refl = (1-fixed_soiling_loss) * plant_design['helio_reflectance'] * 100  # Simulated heliostat reflectivity
            adjust = get_field_adjustment_from_CD_data(year, plant_design['N_hel'], plant_design['helio_reflectance']*100, False, refl, True)

    adjust = adjust.tolist()
    data_steps_per_hour = len(adjust)/8760  
    if data_steps_per_hour != steps_per_hour:
        adjust = translate_to_new_timestep(adjust, 1./data_steps_per_hour, 1./steps_per_hour)
    return adjust


# Get adjusted field availability from CD data (optionally) including soiling loss, heliostats not tracking, and heliostats offline
# Note clean reflectivity is 0 - 100
def get_field_adjustment_from_CD_data(year, nhel, clean_refl, use_CD_soiling = True, specified_refl = None, use_tracking_and_offline = False):

    offline_file = './model-validation/input_files/daily_helios_offline_%d.csv'%year  # Daily 
    tracking_file = './model-validation/input_files/tracking_from_op_report_%d_1min.csv'%year  # Tracking fractions from daily op reports (when available), or HFCS logs 
    tracking_file_HFCS = './model-validation/input_files/tracking_%d_1min.csv'%year   # Tracking fractions from HFCS logs
    
    nph = 60
    soil_avail_per_day = get_CD_soiling_availability(year, clean_refl) if use_CD_soiling else (specified_refl / clean_refl) * np.ones(365)  # Daily annual array 
    fract_avail_per_day = 1.0 - np.genfromtxt(offline_file) / nhel if use_tracking_and_offline else np.ones(365)
    
    if use_tracking_and_offline:
        tracking_1 = np.genfromtxt(tracking_file) / 100.  
        tracking_2 = np.genfromtxt(tracking_file_HFCS) / 100.  
        inds = np.where(np.abs(tracking_1 - tracking_2)>1.e-6)[0]
        days = np.unique( (inds/(nph*24)).astype(int))
        fract_avail_per_day[days] = 1.0   # Heliostats offline are already captured in tracking fraction derived from daily op report
        avail = tracking_1 * np.repeat(soil_avail_per_day*fract_avail_per_day, nph*24)
    else:
        avail = np.repeat(soil_avail_per_day * fract_avail_per_day, nph*24)
        
    adjust = (1.0-avail) * 100

    return adjust


def get_CD_soiling_availability(year, clean_refl):
    refl_file = './model-validation/input_files/daily_refl_%d.csv'%year  # Daily measured reflectivity
    avail = np.minimum(1.0, np.genfromtxt(refl_file, delimiter = ',', skip_header = 1)[:,2]/ clean_refl)  # soiled / clean reflectivity
    return avail



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
    if len(data) == 1:
        # A length-one vector has been supplied. Assume constant throughout.
        return [data[0] for _ in dt_var]

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




# Read weather
def read_weather_data(filename):
    weatherdata = get_weather_header(filename)
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), filename), sep=',', skiprows=2, header=0, skipinitialspace = True)  
    weatherdata['year'] = list(df['Year'])
    weatherdata['month'] = list(df['Month'])
    weatherdata['day'] = list(df['Day'])
    weatherdata['hour'] = list(df['Hour'])
    weatherdata['minute'] = list(df['Minute']) if 'Minute' in df.keys() else [0.0 for j in weatherdata['hour']]
    weatherdata['dn'] = list(df['DNI'])
    weatherdata['df'] = list(df['DHI'])
    weatherdata['gh'] = list(df['GHI'])
    weatherdata['wspd'] = list(df['Wspd'])
    weatherdata['tdry'] = list(df['Tdry'])
    weatherdata['rhum'] = list(df['RH'])
    weatherdata['pres'] = list(df['Pres'])
    return weatherdata


def get_weather_header(filename):
    # os.chdir(os.path.dirname(__file__))
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), filename), sep=',', header=0, nrows=1, skipinitialspace = True, engine = 'python')
    header = {}
    header['tz'] = int(df['Time Zone'][0])
    header['elev'] = float(df['Elevation'][0])
    header['lat'] = float(df['Latitude'][0])
    header['lon'] = float(df['Longitude'][0])
    return header
    

def create_empty_weather_data(like_data, nperhour):
    weather = {k:like_data[k] for k in ['tz', 'elev', 'lat', 'lon']}
    year = like_data['year'][0]
    nperday = nperhour*24
    nperyear = nperhour*8760    
    monthday = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  
    weather['year'] = (year * np.ones(nperyear)).tolist()
    weather['month'] = np.repeat(np.arange(1,13), np.array(monthday)*nperday).tolist()
    weather['day'] = np.array([d+1 for m in range(12) for d in range(monthday[m]) for j in range(nperday)]).tolist()
    weather['hour'] = np.tile(np.repeat(np.arange(24), nperhour), 365).tolist()
    weather['minute'] = np.tile(np.arange(0, 60, 60/nperhour), 24*365).tolist()
    for k in ['dn', 'df', 'gh', 'wspd', 'tdry', 'rhum', 'pres']:
        weather[k] = np.zeros(nperyear).tolist()
    return weather
                                


def update_weather_timestep(weatherdata, nperhour):
    nperhour_file = int(len(weatherdata['dn'])/8760)
    if nperhour == nperhour_file:
        return weatherdata  
    newdata = create_empty_weather_data(weatherdata, nperhour)
    for k in ['dn', 'df', 'gh', 'wspd', 'tdry', 'rhum', 'pres']:
        newdata[k] = translate_to_new_timestep(weatherdata[k], 1./nperhour_file, 1./nperhour)
    return newdata
    



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



# Read weather forecast data starting at midnight (UTC), date is a datetime.date object 
def read_weather_forecast(date, offset30 = True): 
    path = os.path.join(os.path.dirname(__file__), 'weather forecasts/Request_for_Tonopah.zip')
    sday = '%d'%date.day if date.day>=10 else '0%d'%date.day
    smonth = '%d'%date.month if date.month>=10 else '0%d'%date.month
    filename = 'Request_for_Tonopah/%d/0000/%d%s%s_Tonopah.csv'%(date.year, date.year, smonth, sday)
    cols = {'dn':5, 'wspd':7, 'wdir':8, 'tdry':9, 'rhum':10}
    zfile = zipfile.ZipFile(path, mode = 'r')
    data = zfile.read(filename).decode('utf-8')
    data = data.split('\n')[1:]
    wdata = {k:[] for k in cols.keys()}
    for j in range(len(data)):
        sep = data[j].split(',')
        if len(sep)>1:
            for k in cols.keys():
                wdata[k].append(float(sep[cols[k]]))

    sep = data[0].split(',')            
    lat = float(sep[2])
    lon = float(sep[3])
    step = 1.0
    duration = len(wdata['dn']) * step
    dtstart = datetime.datetime(date.year, date.month, date.day) 
    if offset30:
        dtstart = dtstart - datetime.timedelta(minutes = 30)  
    dtend = dtstart + datetime.timedelta(minutes=duration*60-step*60)  
    times = pd.date_range(start = dtstart, end = dtend, freq='%.1fmin'%(step*60), tz = 'UTC')   
    solpos = solarposition.get_solarposition(times, lat, lon)     
    zen = solpos['zenith'].values
    wdata['dn'] = [wdata['dn'][j] / cos(min(zen[j], 89)* 3.14159/180) for j in range(len(wdata['dn']))]
     
    return wdata

    
    
# Read initial plant state from CD daily operations report (one-minute data)
# date is a datetime.date object 
def read_CD_data(date, raw_data_direc, processed_data_direc):    
    sday = '%d'%date.day if date.day>=10 else '0%d'%date.day
    smonth = '%d'%date.month if date.month>=10 else '0%d'%date.month    
    pklname =  '%s_%s_%s.pkl'%(date.year, smonth, sday)
    if os.path.exists(processed_data_direc + pklname):
        data = pickle.load(open(processed_data_direc + pklname, "rb"))
    else:
        print('Reading in CD data: ' + str(date))
        fname = 'CDS Daily Operations Report %s %s %s.xlsb'%(date.year, smonth, sday)
        if fname not in os.listdir(raw_data_direc):
            return None
        data = pd.read_excel(raw_data_direc + fname, sheet_name = 'Chart2', engine = 'pyxlsb', header = 3, skiprows=[4], nrows = 1440)
        bad_col_name = "Cold Salt Pmps Disch Flow"
        new_col_names = {}
        is_shift_cols = False
        ### DO NOT change column order between read_excel and this loop 
        for c in data.columns:
            if c == bad_col_name:
                is_shift_cols = True
                prev_col = c
            elif not is_shift_cols or c == 'TOTAL RA SYSTEM FLOW':  # Last column is correct
                new_col_names[c] = c # data and column name are correct
            else:
                new_col_names[prev_col] = c     
                prev_col = c
        data = data.drop(labels = ["Pond Drain Pump 2"], axis = 'columns')        # This column data is lost
        data = data.rename(columns = new_col_names)
        data = data.set_index('Date')
        pickle.dump(data, open(processed_data_direc + pklname, "wb") )
    return data

def get_clean_CD_cycle_data(data, inter_missing_data = True, date_w_ctrl_adj = False, flow_avg_wind = 20, des_cold_temp = 295.0):
    # Cleans CD data for target setting

    # Interpolating missing data 
    data = data.interpolate(method='quadratic')

    D = pd.DataFrame()
    D['Gross Power [MW]'] = data['Gen Gross Active Power']
    D['Hot Tank Temp [C]'] = (data['Hot Salt Tank Standpipe Av Tmp'] - 32.)*5./9.         # F -> C
    D['Cold Tank Temp [C]'] = (data['Cold Salt Tank Standpipe Av Tmp'] - 32.)*5./9.        # F -> C
    D['Hot Tank Level [m]'] = data['Hot Salt Tank Level Seltd']/3.28084                    # ft -> m
    D['Cold Tank Level [m]'] = data['Col Salt Tank Level Seltd']/3.28084                   # ft -> m

    # Cycle return temperature - cleaning data
    salt_outs = ['Econ 1 Salt Out Tmp', 'Econ 2 Salt Out Tmp']
    for s_out_temp in salt_outs:
        data.loc[(data[s_out_temp]<0.001), s_out_temp] = np.NaN # replacing all 0 with NaN (bad data) (NOTE: temperature never is close to 0 in this data)
        data[s_out_temp].interpolate()

    # Filtering data to remove the train out of operation from average calc.
    D['Cycle Return Temp [F]'] = 0.5*(data['Econ 1 Salt Out Tmp'] + data['Econ 2 Salt Out Tmp'])
    mask = ((data['Econ 1 Salt Out Tmp'] - data['Econ 2 Salt Out Tmp']) > 5.)
    D.loc[mask, 'Cycle Return Temp [F]'] = data['Econ 1 Salt Out Tmp']           # Note: this could be a little bit better
    mask = ((data['Econ 2 Salt Out Tmp'] - data['Econ 1 Salt Out Tmp']) > 5.)
    D.loc[mask, 'Cycle Return Temp [F]'] = data['Econ 2 Salt Out Tmp']
    D['Cycle Return Temp [C]'] = (D['Cycle Return Temp [F]'] - 32.)*5./9.        # F -> C

    # CD tank conditions -> salt mass in tank
    tank_radius = 21.336    # [m]   Assumed radius based on PID schematic
    dens_h = salt_density(D['Hot Tank Temp [C]']+273.15)
    dens_c = salt_density(D['Cold Tank Temp [C]']+273.15)
    D['Hot Tank Mass [kg]'] = dens_h*D['Hot Tank Level [m]']*np.pi*(tank_radius**2)
    D['Cold Tank Mass [kg]'] = dens_c*D['Cold Tank Level [m]']*np.pi*(tank_radius**2)

    # Flow into cycle
    D['Flow into cycle [kg/s]'] = (data['Superheater 1 Flow'] + data['Reheater 1 Flow'] 
                                            + data['Superheater 2 Flow'] + data['Reheater 2 Flow'])/2.20462       # lbs/s -> kg/s
    # Determine if trains are operational
    pwr = data['Gen Gross Active Power']/max(data['Gen Gross Active Power'])
    t1_flow = (data['Superheater 1 Flow'] + data['Reheater 1 Flow'])/max(D['Flow into cycle [kg/s]']) # avg 
    t2_flow = (data['Superheater 2 Flow'] + data['Reheater 2 Flow'])/max(D['Flow into cycle [kg/s]'])

    # TODO: These limits are arbitrary and may not work outside of data observed
    if date_w_ctrl_adj:
        pow_lim = 0.25  # generating power greater than 25%
    else:
        pow_lim = 0.01  # generating power greater than 1%

    D.loc[pwr <= pow_lim, 'Flow into cycle [kg/s]'] = 0.0
    mask_t1 = (pwr > pow_lim) & ((t1_flow - t2_flow) > 0.03)    # generating power greater than power limit, train 1 flow is 3% greater then train 2
    D.loc[mask_t1, 'Flow into cycle [kg/s]'] = (data['Superheater 1 Flow'] + data['Reheater 1 Flow']).loc[mask_t1]/2.20462  # Train 1 only
    mask_t2 = (pwr > pow_lim) & ((t2_flow - t1_flow) > 0.03)
    D.loc[mask_t2, 'Flow into cycle [kg/s]'] = (data['Superheater 2 Flow'] + data['Reheater 2 Flow']).loc[mask_t2]/2.20462  # Train 2 only

    # Rolling avgerage to set flow rate
    df = D['Flow into cycle [kg/s]'].expanding().mean()
    df2 = D['Flow into cycle [kg/s]'].rolling(window=flow_avg_wind).mean()  # TODO: average over center (center=True)
    D['Avg flow into cycle [kg/s]'] = pd.concat([df.iloc[0:flow_avg_wind], df2.iloc[flow_avg_wind:]])

    Tavg = 0.5*(D['Hot Tank Temp [C]'] + D['Cycle Return Temp [C]']) + 273.15 # update average temp
    cavg = salt_cp(Tavg)
    
    D['Q into cycle [MW]'] = D['Flow into cycle [kg/s]']*cavg*(D['Hot Tank Temp [C]'] - D['Cycle Return Temp [C]'])*1.e-6
    D['Avg Q into cycle [MW]'] = D['Avg flow into cycle [kg/s]']*cavg*(D['Hot Tank Temp [C]'] - D['Cycle Return Temp [C]'])*1.e-6   

    # TODO: Assuming 1 meter of dead space based on PID schmatic and CD data -> subtract dead mass
    D['E charge TES [MWht]'] = (D['Hot Tank Mass [kg]'] - dens_h*np.pi*(tank_radius**2))*cavg*(D['Hot Tank Temp [C]'] - des_cold_temp)*1.e-6/(3600.)    # J -> MJ -> MWh  # TODO: bring in design value
    D['Net Power [MW]'] = data['Unit Net MW']
    
    
    #mflow1 = data['RA Line1 Flow']
    #mflow2 = data['RA Line2 Flow']
    #Tout1 = (data['Slctd Rcvr Ckt 1 Outlet '] - 32.)*5./9. 
    #Tout2 = (data['Receiver Ckt 2 Outlet'] - 32.)*5./9. 
    #D['Rec avg Tout [C]'] =  (mflow1*Tout1 + mflow2*Tout2)/np.maximum((mflow1 + mflow2), 1.e-6)
    D['Rec avg Tout [C]'] = (data['Downcomer Hot Salt Tmp Slctd']- 32.)*5./9. 
    return D
        


# Read in NVE-provided dispatch schedule from CD data files
def read_NVE_schedule(date, raw_data_direc):
    sday = '%d'%date.day if date.day>=10 else '0%d'%date.day
    smonth = '%d'%date.month if date.month>=10 else '0%d'%date.month
    fname = 'CDS Daily Operations Report %s %s %s.xlsb'%(date.year, smonth, sday)
    if fname not in os.listdir(raw_data_direc):
        return None
    data = pd.read_excel(raw_data_direc + fname, sheet_name = 'NVE - Day Ahead', engine = 'pyxlsb', usecols = [2,3,4,5], skiprows= 24)
    targets = data['Plant Total'].values[1:-1].astype(float)  # Net generation 
    return targets








# Get plant state at 12am standard time (1am local time if during DST)
def get_initial_state_from_CD_data(date, raw_data_direc, processed_data_direc, design):
    state = plant.plant_initial_state_CD
    
    data = read_CD_data(date, raw_data_direc, processed_data_direc)
    if data is None:
        return  None
    
    i = 0 if not is_dst(date) else 60  # Row from which to extract initial state.  If this is during DST, start simulation a 1am (12am standard time)
    
    # Field / receiver
    state.is_field_tracking_init = 1 if data['Heliostats At Track'].values[i] > 0 else 0
    state.rec_op_mode_initial = 0  # Off
    if state.is_field_tracking_init == 1:
        state.rec_op_mode_initial = 2  # On
    elif data['Heliostats At Preheat'].values[i] > 0:
        state.rec_op_mode_initial = 1 # Startup
    
    # TES
    hot_level = float(data['Hot Salt Tank Level Seltd'].values[i])/3.28084  #m   
    state.T_tank_cold_init = (float(data['Cold Salt Tank Standpipe Av Tmp'].values[i]) - 32. )*5./9.  # C
    state.T_tank_hot_init = (float(data['Hot Salt Tank Standpipe Av Tmp'].values[i]) - 32. )*5./9.    # C
    state.csp_pt_tes_init_hot_htf_percent = (hot_level - design.h_tank_min) / (design.h_tank - design.h_tank_min) * 100

    # Cycle
    state.wdot0 = max(0.0, float(data['Gen Gross Active Power'].values[i]))
    state.pc_op_mode_initial = 3 # Off
    if state.wdot0 > 0.05*design.P_ref:
        state.pc_op_mode_initial = 1  # On
        
    return state

# Salt heat capacity and density
def salt_cp(T):  # T in K
    return 1000*(1.4387 + 5e-6*T + 2e-7*T*T - 1e-10*T*T*T)  

def salt_density(T): # T in K
    return 2299.4-0.7875*(T) + 2.e-4*(T**2) - 1.e-7*(T**3)


def rename_dict_keys(dictionary, key_map):
    """
    Renames in place the keys in dictionary using the key_map. May not preserve dict order.

    key_map -   keys are starting names, values are ending names
    """
    for k,v in key_map.items():
        try:
            dictionary[v] = dictionary.pop(k)
        except:
            pass
    
    return


def rename_dict_keys_reversed(dictionary, key_map):
    """
    Renames in place the keys in dictionary using the key_map, reverse convention. May not preserve dict order.

    key_map -   keys are ending names, values are starting names
    """
    for k,v in key_map.items():
        try:
            dictionary[k] = dictionary.pop(v)
        except:
            pass
    
    return


# inv_map = {v: k for k, v in my_map.items()}

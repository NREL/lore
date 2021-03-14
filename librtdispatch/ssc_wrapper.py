import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import PySSC as api
import copy
import numpy as np
import timeit

from mspt_2020_defaults import vartab as V


#-------------------------------------------------------------------------
# Update incompatible names in ssc
#  (only used in call_ssc)
def update_bad_names(D):
    name_map = { # Name used in class               # Name used in ssc
                'csp_pt_rec_max_oper_frac':        'csp.pt.rec.max_oper_frac',
                'csp_pt_tes_init_hot_htf_percent': 'csp.pt.tes.init_hot_htf_percent'
                }
    
    for k in name_map:
        if k in D.keys():
            D[name_map[k]] = D[k]
    return


#-------------------------------------------------------------------------
# Utility functions to pass information to/from ssc
# Set ssc input parameters from data in Dict
#  (only used in call_ssc)
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
            elif type(Dict[key]) == type({}):
                table = ssc_api.data_create()
                set_ssc_data_from_dict(ssc_api, table, Dict[key])
                ssc_api.data_set_table(ssc_data, key.encode("utf-8"), table)
                ssc_api.data_free(table)
            else:
               print ("Could not assign variable " + key )
               raise KeyError
        except:
            print ("Error assigning variable " + key + ": bad data type")


# Run ssc simulation using user-defined input parameters in dictionary D (all other input parameters will be set to default values).
def call_ssc(D, retvars = ['gen'], plant_state_pt = -1, npts = None):
    """
    Call SSC via PySSC

    Inputs:
        D               dict of SSC inputs
        retvars         names of SSC outputs to return from this function
        plant_state_pt  time point at which to save plant state
        npts            number of points in each array to save; None saves all

    Outputs:
        R               dict of select SSC outputs per those chosen by retvars
        plant_state     Plant.state dictionary
    """

    ssc = api.PySSC()
    dat = ssc.data_create()
    mspt = ssc.module_create("tcsmolten_salt".encode("utf-8"))	
       
    update_bad_names(D)  # Update keys in D that were changed from ssc variable names
    Vt = copy.deepcopy(V)
    Vt.update(D)
    set_ssc_data_from_dict(ssc, dat, Vt)

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
            if npts is not None:  
                R[k] = R[k][0:npts]
            R[k] = np.array(R[k])
                
    # Save plant state at designated time point
    plant_state = set_from_ssc(ssc, dat, plant_state_pt)
    
    ssc.module_free(mspt)
    ssc.data_free(dat)   
    
    return R, plant_state


# Set plant state from ssc data structure using conditions at time index t (relative to start of simulation)
def set_from_ssc(sscapi, sscdata, t):
    # Plant state input/output variable name map (from pysam_wrap.py in LORE/loredash/mediation)
    plant_state_io_map = { # Number Inputs                         # Arrays Outputs
                        'pc_op_mode_initial':                   'pc_op_mode_final',
                        'pc_startup_time_remain_init':          'pc_startup_time_remain_final',
                        'pc_startup_energy_remain_initial':     'pc_startup_energy_remain_final',
                        'is_field_tracking_init':               'is_field_tracking_final',
                        'rec_op_mode_initial':                  'rec_op_mode_final',
                        'rec_startup_time_remain_init':         'rec_startup_time_remain_final',
                        'rec_startup_energy_remain_init':       'rec_startup_energy_remain_final',
                        'T_tank_hot_init':                      'T_tes_hot',
                        'T_tank_cold_init':                     'T_tes_cold',
                        'csp_pt_tes_init_hot_htf_percent':      'hot_tank_htf_percent_final',       # in SSC this variable is named csp.pt.tes.init_hot_htf_percent
                        
                        # Variables for dispatch model (note these are not inputs for ssc)
                        # Number Inputs for dispatch,            # Array outputs 
                        'wdot0':                                 'P_cycle',  # TODO: Output arrays for P_cycle and q_pb aretime-step averages. Should create new output in ssc for value at end of timestep -  but not very important for short timesteps used here
                        'qdot0':                                 'q_pb',
                        }
    
    state = {}
    for k in plant_state_io_map.keys():
        kout = plant_state_io_map[k]
        array = sscapi.data_get_array(sscdata, kout.encode('utf-8'))
        state[k] = array[t]
    
    return state


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
        R, state = call_ssc(D, ['eta_map_out', 'flux_maps_for_import', 'A_sf'])
        print('Time to simulate flux maps = %.2fs'%(timeit.default_timer() - start))
        # return flux_maps
        A_sf_in = R['A_sf']
        eta_map = R['eta_map_out']
        flux_maps = [x[2:] for x in R['flux_maps_for_import']]
        return {'A_sf_in': A_sf_in, 'eta_map': eta_map, 'flux_maps': flux_maps}


def estimates_for_dispatch_model(plant_design, toy, horizon, weather_data, N_pts_horizon, clearsky_data, start_pt):
    """
    Outputs:
        ssc_outputs         a 7-item dictionary of arrays, selected according to retvars
    """

    D_est = plant_design.copy()
    D_est['time_stop'] = toy + horizon
    D_est['is_dispatch_targets'] = False
    D_est['tshours'] = 100                      # Inflate TES size so that there is always "somewhere" to put receiver output
    D_est['solar_resource_data'] = weather_data
    D_est['is_rec_startup_trans'] = False
    D_est['rec_su_delay'] = 0.001               # Simulate with no start-up time to get total available solar energy
    D_est['rec_qf_delay'] = 0.001
    retvars = ['Q_thermal', 'm_dot_rec', 'beam', 'clearsky', 'tdry', 'P_tower_pump', 'pparasi']

    ssc_outputs, new_state = call_ssc(D_est, retvars, npts = N_pts_horizon)
    if ssc_outputs['clearsky'].max() < 1.e-3:         # Clear-sky data wasn't passed through ssc (ssc controlled from actual DNI, or user-defined flow inputs)
        ssc_outputs['clearsky'] = clearsky_data[start_pt : start_pt + N_pts_horizon]

    return ssc_outputs
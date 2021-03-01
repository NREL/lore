import PySSC as api
import copy
import numpy as np

import util
import dispatch
from mspt_2020_defaults import vartab as V


class PlantState:
    def __init__(self):
        
        # Field and receiver
        self.is_field_tracking_init = 0             # Is field tracking?
        self.rec_op_mode_initial = 0                # Receiver operating mode (0 = off, 1 = startup, 2 = on)
        self.rec_startup_time_remain_init = 0.0     # Receiver startup time remaining (hr)
        self.rec_startup_energy_remain_init = 0.0   # Receiver startup energy remaining (Wh)
        self.disp_rec_persist0 = 0.0                # Time (hr) that receiver has been in its current state
        self.disp_rec_off0 = 0.0                    # Time (hr) that receiver has not been operating (off or startup)
        
        # TES
        self.T_tank_cold_init = 290.0                # Cold tank temperature (C)
        self.T_tank_hot_init = 565.0                 # Hot tank temperature (C)
        self.csp_pt_tes_init_hot_htf_percent = 30.0  # Fraction of available storage in hot tank (%)
        
        # Cycle
        self.pc_op_mode_initial = 3                 # Initial cycle operating mode (0 = startup, 1 = on, 2 = standby, 3 = off, 4 = startup_controlled)
        self.pc_startup_time_remain_init = 0.0       # Cycle startup time remaining (hr)
        self.pc_startup_energy_remain_initial = 0.0  # Cycle startup energy remaining (kWh)
        self.wdot0 = 0.0                             # Cycle electricity generation (MWe)
        self.qdot0 = 0.0                             # Cycle thermal input (MWt)
        self.disp_pc_persist0 = 0.0                  # Time (hr) that cycle has been in its current state
        self.disp_pc_off0 = 0.0                      # Time (hr) that cycle has not been generating electric power (off, startup, or standby)

        return


    def set_default(self, design, properties):
        self.is_field_tracking_init = 0             
        self.rec_op_mode_initial = 0                
        self.rec_startup_time_remain_init = properties.rec_su_delay    
        self.rec_startup_energy_remain_init = properties.rec_qf_delay*design.Qrec * 1.e6  
        self.disp_rec_persist0 = 1000   # Set to be non-binding in dispatch model
        self.disp_rec_off0 = 1000       # Set to be non-binding in dispatch model
        
        self.T_tank_cold_init = design.T_htf_cold_des               
        self.T_tank_hot_init = design.T_htf_hot_des               
        self.csp_pt_tes_init_hot_htf_percent = 30.0  

        self.pc_op_mode_initial = 3                 
        self.pc_startup_time_remain_init  = properties.startup_time
        self.pc_startup_energy_remain_initial = properties.startup_frac * design.get_cycle_thermal_rating() * 1000.
        self.wdot0 = 0.0
        self.qdot0 = 0.0
        self.disp_pc_persist0 = 1000    # Set to be non-binding in dispatch model
        self.disp_pc_off0 = 1000       # Set to be non-binding in dispatch model
        
        return
        
    
    # Set plant state from ssc data structure using conditions at time index t (relative to start of simulation)
    def set_from_ssc(self, sscapi, sscdata, t):

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
        
        for k in plant_state_io_map.keys():
            kout = plant_state_io_map[k]
            array = sscapi.data_get_array(sscdata, kout.encode('utf-8'))
            setattr(self, k, array[t])
        return
    
    


class FluxMaps:
    def __init__(self):
        self.A_sf_in = 0.0
        self.eta_map = []
        self.flux_maps = []
        return
    
    def set_from_ssc(self, S):
        # S = dictionary of ssc ouputs
        self.A_sf_in = S['A_sf']
        self.eta_map = S['eta_map_out']
        self.flux_maps = [x[2:] for x in S['flux_maps_for_import']]
        return


#-------------------------------------------------------------------------
# Update incompatible names in ssc
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
        plant_state     PlantState object representing the final plant state
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
    plant_state = PlantState()
    plant_state.set_from_ssc(ssc, dat, plant_state_pt)
    
    ssc.module_free(mspt)
    ssc.data_free(dat)   
    
    return R, plant_state



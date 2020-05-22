# -*- coding: utf-8 -*-
"""
Pyomo real-time dispatch model

"""
#import pyomo
import pyomo.environ as pe


class RealTimeDispatchModel(object):
    def __init__(self, params, include={"pv":False,"battery":False,"persistence":False}):
        self.model = pe.ConcreteModel()
        self.include = include
        self.generateParams(params)
        self.generateVariables()
        self.addObjective()
        self.generateConstraints()

    def generateParams(self,params):
        ### Sets and Indices ###
        self.model.T = pe.Set(initialize = range(1,params["num_periods"]+1))  #T: time periods
        #------- Time-indexed parameters --------------
        self.model.Delta = pe.Param(self.model.T, mutable=True, initialize=0)          #duration of period t
        self.model.Delta_e = pe.Param(self.model.T, mutable=True, initialize=0)       #cumulative time elapsed at end of period t
        ### Time-series CSP Parameters ##
        self.model.delta_rs = pe.Param(self.model.T, mutable=True, initialize=0) # \delta^{rs}_{t}: Estimated fraction of period $t$ required for receiver start-up [-]
        self.model.D = pe.Param(self.model.T, mutable=True, initialize=0) #D_{t}: Time-weighted discount factor in period $t$ [-]
        self.model.etaamb = pe.Param(self.model.T, mutable=True, initialize=0)  #\eta^{amb}_{t}: Cycle efficiency ambient temperature adjustment factor in period $t$ [-]
        self.model.etac = pe.Param(self.model.T, mutable=True, initialize=0)   #\eta^{c}_{t}:   Normalized condenser parasitic loss in period $t$ [-] 
        self.model.P = pe.Param(self.model.T, mutable=True, initialize=0)       #P_{t}: Electricity sales price in period $t$ [\$/kWh\sse]
        self.model.Qin = pe.Param(self.model.T, mutable=True, initialize=0)    #Q^{in}_{t}: Available thermal power generated by the CSP heliostat field in period $t$ [kW\sst]
        self.model.Qc = pe.Param(self.model.T, mutable=True, initialize=0)     #Q^{c}_{t}: Allowable power per period for cycle start-up in period $t$ [kW\sst]
        self.model.Wdotnet = pe.Param(self.model.T, mutable=True, initialize=0)  #\dot{W}^{net}_{t}: Net grid transmission upper limit in period $t$ [kW\sse]
        self.model.W_u_plus = pe.Param(self.model.T, mutable=True, initialize=0)  #W^{u+}_{t}: Maximum power production when starting generation in period $t$  [kW\sse]
        self.model.W_u_minus = pe.Param(self.model.T, mutable=True, initialize=0)  #W^{u-}_{t}: Maximum power production in period $t$ when stopping generation in period $t+1$  [kW\sse]
        
        ### Time-Series PV Parameters ###
        if self.include["pv"]:
            self.model.w_pv = pe.Param(self.model.T, mutable=True, initialize=0)      #w^{PV}_t: <aximum DC power production from PV system in period $t$
        
        ###  Cost Parameters ###
        self.model.alpha = pe.Param(mutable=True, initialize=0)        #alpha: Conversion factor between unitless and monetary values [\$]
        self.model.Crec = pe.Param(mutable=True, initialize=0)         #Crec: Operating cost of heliostat field and receiver [\$/kWh\sst]
        self.model.Crsu = pe.Param(mutable=True, initialize=0)         #Crsu: Penalty for receiver cold start-up [\$/start]
        self.model.Crhsp = pe.Param(mutable=True, initialize=0)        #Crhsp: Penalty for receiver hot start-up [\$/start]
        self.model.Cpc = pe.Param(mutable=True, initialize=0)          #Cpc: Operating cost of power cycle [\$/kWh\sse]
        self.model.Ccsu = pe.Param(mutable=True, initialize=0)        #Ccsu: Penalty for power cycle cold start-up [\$/start]
        self.model.Cchsp = pe.Param(mutable=True, initialize=0)       #Cchsp: Penalty for power cycle hot start-up [\$/start]
        self.model.C_delta_w = pe.Param(mutable=True, initialize=0)    #C_delta_w: Penalty for change in power cycle  production [\$/$\Delta\text{kW}$\sse]
        self.model.C_v_w = pe.Param(mutable=True, initialize=0)        #C_v_w: Penalty for change in power cycle  production \tcb{beyond designed limits} [\$/$\Delta\text{kW}$\sse]
        self.model.Ccsb = pe.Param(mutable=True, initialize=0)         #Ccsb: Operating cost of power cycle standby operation [\$/kWh\sst]
        
        # -------PV and Battery Cost Parameters -------
        if self.include["pv"]:
            self.model.Cpv = pe.Param(mutable=True, initialize=0)    #Operating cost of photovoltaic field [\$/kWh\sse]
        if self.include["battery"]:
            self.model.Cbc = pe.Param(mutable=True, initialize=0)    #Operating cost of charging battery [\$/kWh\sse]
            self.model.Cbd = pe.Param(mutable=True, initialize=0)    #Operating cost of discharging battery [\$/kWh\sse]
            self.model.Cbl = pe.Param(mutable=True, initialize=0)    #Lifecycle cost for battery [\$/lifecycle]
        
        ### CSP Field and Receiver Parameters ###
        self.model.deltal = pe.Param(mutable=True, initialize=0)    #Minimum time to start the receiver [hr]
        self.model.Ehs = pe.Param(mutable=True, initialize=0)       #Heliostat field startup or shut down parasitic loss [kWh\sse]
        self.model.Er = pe.Param(mutable=True, initialize=0)        #Required energy expended to start receiver [kWh\sst]
        self.model.Eu = pe.Param(mutable=True, initialize=0)        #Thermal energy storage capacity [kWh\sst]
        self.model.Lr = pe.Param(mutable=True, initialize=0)        #Receiver pumping power per unit power produced [kW\sse/kW\sst]
        self.model.Qrl = pe.Param(mutable=True, initialize=0)       #Minimum operational thermal power delivered by receiver [kWh\sst]
        self.model.Qrsb = pe.Param(mutable=True, initialize=0)      #Required thermal power for receiver standby [kWh\sst]
        self.model.Qrsd = pe.Param(mutable=True, initialize=0)      #Required thermal power for receiver shut down [kWh\sst] 
        self.model.Qru = pe.Param(mutable=True, initialize=0)       #Allowable power per period for receiver start-up [kWh\sst]
        self.model.Wh = pe.Param(mutable=True, initialize=0)        #Heliostat field tracking parasitic loss [kW\sse]
        self.model.Wht = pe.Param(mutable=True, initialize=0)       #[az] this isn't in the implementation.  Tower piping heat trace parasitic loss [kW\sse]
        
        ### Power Cycle Parameters ###
        self.model.Ec = pe.Param(mutable=True, initialize=0)           #Required energy expended to start cycle [kWh\sst]
        self.model.eta_des = pe.Param(mutable=True, initialize=0)      #Cycle nominal efficiency [-] 
        self.model.etap = pe.Param(mutable=True, initialize=0)         #Slope of linear approximation of power cycle performance curve [kW\sse/kW\sst]
        self.model.Lc = pe.Param(mutable=True, initialize=0)           #Cycle heat transfer fluid pumping power per unit energy expended [kW\sse/kW\sst]
        self.model.Qb = pe.Param(mutable=True, initialize=0)           #Cycle standby thermal power consumption per period [kW\sst]
        self.model.Ql = pe.Param(mutable=True, initialize=0)           #Minimum operational thermal power input to cycle [kW\sst]
        self.model.Qu = pe.Param(mutable=True, initialize=0)           #Cycle thermal power capacity [kW\sst]
        self.model.Wb = pe.Param(mutable=True, initialize=0)           #Power cycle standby operation parasitic load [kW\sse]
        self.model.Wdotl = pe.Param(mutable=True, initialize=0)        #Minimum cycle electric power output [kW\sse]
        self.model.Wdotu = pe.Param(mutable=True, initialize=0)        #Cycle electric power rated capacity [kW\sse]
        self.model.W_delta_plus = pe.Param(mutable=True, initialize=0) #Power cycle ramp-up designed limit [kW\sse/h]
        self.model.W_delta_minus = pe.Param(mutable=True, initialize=0)#Power cycle ramp-down designed limit [kW\sse/h]
        self.model.W_v_plus = pe.Param(mutable=True, initialize=0)     #Power cycle ramp-up violation limit [kW\sse/h]
        self.model.W_v_minus = pe.Param(mutable=True, initialize=0)    #Power cycle ramp-down violation limit [kW\sse/h]
        self.model.Yu = pe.Param(mutable=True, initialize=0)           #Minimum required power cycle uptime [h]
        self.model.Yd = pe.Param(mutable=True, initialize=0)           #Minimum required power cycle downtime [h]
        
        ### Initial Condition Parameters ###
        self.model.s0 = pe.Param(mutable=True, initialize=0)  #Initial TES reserve quantity  [kWh\sst]
        self.model.ucsu0 = pe.Param(mutable=True, initialize=0) #Initial cycle start-up energy inventory  [kWh\sst]
        self.model.ursu0 = pe.Param(mutable=True, initialize=0) #Initial receiver start-up energy inventory [kWh\sst]
        self.model.wdot0 = pe.Param(mutable=True, initialize=0) #Initial power cycle electricity generation [kW\sse]
        self.model.yr0 = pe.Param(mutable=True, initialize=0)  #1 if receiver is generating ``usable'' thermal power initially = pe.Param(mutable=True, initialize=0) 0 otherwise  [az] this is new.
        self.model.yrsb0 = pe.Param(mutable=True, initialize=0)  #1 if receiver is in standby mode initially = pe.Param(mutable=True, initialize=0) 0 otherwise  [az] this is new.
        self.model.yrsu0 = pe.Param(mutable=True, initialize=0)  #1 if receiver is in starting up initially = pe.Param(mutable=True, initialize=0) 0 otherwise    [az] this is new.
        self.model.y0 = pe.Param(mutable=True, initialize=0)  #1 if cycle is generating electric power initially = pe.Param(mutable=True, initialize=0) 0 otherwise
        self.model.ycsb0 = pe.Param(mutable=True, initialize=0)  #1 if cycle is in standby mode initially = pe.Param(mutable=True, initialize=0) 0 otherwise
        self.model.ycsu0 = pe.Param(mutable=True, initialize=0)  #1 if cycle is in starting up initially = pe.Param(mutable=True, initialize=0) 0 otherwise    [az] this is new.
        self.model.Yu0 = pe.Param(mutable=True, initialize=0)  # duration that cycle has been generating electric power [h]
        self.model.Yd0 = pe.Param(mutable=True, initialize=0)  # duration that cycle has not been generating power (i.e., shut down or in standby mode) [h]
        
        # -------Persistence Parameters ---------
        if self.include["persistence"]:
            self.model.wdot_s_prev  = pe.Param(self.model.T, mutable=True, initialize=0)
            self.model.wdot_s_pen  = pe.Param(self.model.T, mutable=True, initialize=0)
        
        # -------Miscellaneous Parameters taken from SAM---------
        self.model.day_of_year = pe.Param(mutable=True, initialize=0)
        self.model.disp_time_weighting = pe.Param(mutable=True, initialize=0)
        self.model.csu_cost = pe.Param(mutable=True, initialize=0)
        self.model.eta_cycle = pe.Param(mutable=True, initialize=0)
        self.model.gamma = pe.Param(mutable=True, initialize=0)
        self.model.gammac = pe.Param(mutable=True, initialize=0)
        self.model.M = pe.Param(mutable=True, initialize=0) 
        self.model.qrecmaxobs = pe.Param(mutable=True, initialize=0)
        self.model.W_dot_cycle = pe.Param(mutable=True, initialize=0)
        self.model.Z_1 = pe.Param(mutable=True, initialize=0)
        self.model.Z_2 = pe.Param(mutable=True, initialize=0)
        self.model.max_up = pe.Param(mutable=True, initialize=0)
        self.model.max_down = pe.Param(mutable=True, initialize=0)
        self.model.max_up_v = pe.Param(mutable=True, initialize=0)
        self.model.max_down_v = pe.Param(mutable=True, initialize=0)
        self.model.pen_delta_w = pe.Param(mutable=True, initialize=0)
        self.model.q0 = pe.Param(mutable=True, initialize=0)
        self.model.rsu_cost = pe.Param(mutable=True, initialize=0)
        self.model.tdown0 = pe.Param(mutable=True, initialize=0)
        self.model.tstby0 = pe.Param(mutable=True, initialize=0)
        self.model.tup0 = pe.Param(mutable=True, initialize=0)
        self.model.Wdot0 = pe.Param(mutable=True, initialize=0)
        self.model.wnet_lim_min = pe.Param(self.model.T, mutable=True, initialize=0)
        self.model.cap_frac = pe.Param(self.model.T, mutable=True, initialize=0)
        self.model.eff_frac = pe.Param(self.model.T, mutable=True, initialize=0)
        self.model.dt = pe.Param(self.model.T, mutable=True, initialize=0)
        self.model.dte = pe.Param(self.model.T, mutable=True, initialize=0)
        self.model.twt = pe.Param(self.model.T, mutable=True, initialize=0)
        
        
        #--------Parameters for the Battery---------
        if self.include["battery"]:
            self.model.alpha_p = pe.Param(mutable=True, initialize=0)    #Bi-directional converter slope-intercept parameter
            self.model.alpha_n = pe.Param(mutable=True, initialize=0)	  #Bi-directional converter slope-intercept parameter
            self.model.beta_p = pe.Param(mutable=True, initialize=0)     #Bi-directional converter slope parameter
            self.model.beta_n = pe.Param(mutable=True, initialize=0)	  #Bi-directional converter slope parameter
            self.model.C_B = pe.Param(mutable=True, initialize=0)
            self.model.C_p = pe.Param(mutable=True, initialize=0)
            self.model.C_n = pe.Param(mutable=True, initialize=0)
            self.model.I_upper_p = pe.Param(mutable=True, initialize=0)
            self.model.I_upper_n = pe.Param(mutable=True, initialize=0)  #Battery discharge current max
            self.model.S_B_lower = pe.Param(mutable=True, initialize=0)
            self.model.S_B_upper = pe.Param(mutable=True, initialize=0)
            self.model.I_lower_n = pe.Param(mutable=True, initialize=0)
            self.model.I_lower_p = pe.Param(mutable=True, initialize=0)
            self.model.P_B_lower = pe.Param(mutable=True, initialize=0)
            self.model.P_B_upper = pe.Param(mutable=True, initialize=0)  #Battery min/max power rating
            self.model.A_V = pe.Param(mutable=True, initialize=0)
            self.model.B_V = pe.Param(mutable=True, initialize=0)	  #Battery linear voltage model slope/intercept coeffs
            self.model.R_int = pe.Param(mutable=True, initialize=0)
            self.model.I_avg = pe.Param(mutable=True, initialize=0)	  #Typical current expected from the battery
            self.model.alpha_pv = pe.Param(mutable=True, initialize=0)
            self.model.beta_pv = pe.Param(mutable=True, initialize=0)
            self.model.Winv_lim = pe.Param(mutable=True, initialize=0)	  # Inverter max power (DC)
            self.model.Wmax = pe.Param(mutable=True, initialize=0)	  #Constant Max power to grid
            self.model.Winvnt = pe.Param(mutable=True, initialize=0)
            self.model.N_csp = pe.Param(mutable=True, initialize=0)

    def generateVariables(self):
        ### Decision Variables ###
        ##--------- Variables ------------------------
        self.model.s = pe.Var(self.model.T, domain=pe.NonNegativeReals, bounds = (0,self.model.Eu))                      #TES reserve quantity at period $t$  [kWh\sst]
        self.model.ucsu = pe.Var(self.model.T, domain=pe.NonNegativeReals)                         #Cycle start-up energy inventory at period $t$ [kWh\sst]
        self.model.ursu = pe.Var(self.model.T, domain=pe.NonNegativeReals)                         #Receiver start-up energy inventory at period $t$ [kWh\sst]
        self.model.wdot = pe.Var(self.model.T, domain=pe.NonNegativeReals)                         #Power cycle electricity generation at period $t$ [kW\sse]
        self.model.wdot_delta_plus = pe.Var(self.model.T, domain=pe.NonNegativeReals)	             #Power cycle ramp-up in period $t$ [kW\sse]
        self.model.wdot_delta_minus = pe.Var(self.model.T, domain=pe.NonNegativeReals)	         #Power cycle ramp-down in period $t$ [kW\sse]
        self.model.wdot_v_plus = pe.Var(self.model.T, domain=pe.NonNegativeReals, bounds = (0,self.model.W_v_plus))      #Power cycle ramp-up beyond designed limit in period $t$ [kW\sse]
        self.model.wdot_v_minus = pe.Var(self.model.T, domain=pe.NonNegativeReals, bounds = (0,self.model.W_v_minus)) 	 #Power cycle ramp-down beyond designed limit in period $t$ [kW\sse]
        self.model.wdot_s = pe.Var(self.model.T, domain=pe.NonNegativeReals)	                     #Energy sold to grid in time t
        self.model.wdot_p = pe.Var(self.model.T, domain=pe.NonNegativeReals)	                     #Energy purchased from the grid in time t
        self.model.x = pe.Var(self.model.T, domain=pe.NonNegativeReals)                            #Cycle thermal power utilization at period $t$ [kW\sst]
        self.model.xr = pe.Var(self.model.T, domain=pe.NonNegativeReals)	                         #Thermal power delivered by the receiver at period $t$ [kW\sst]
        self.model.xrsu = pe.Var(self.model.T, domain=pe.NonNegativeReals)                         #Receiver start-up power consumption at period $t$ [kW\sst]
        
        
        #----------Continuous for the Battery-----------
        if self.include["battery"]:
            self.model.soc = pe.Var(self.model.T, domain=pe.NonNegativeReals)	    #State of charge of battery in time t
            self.model.wbd = pe.Var(self.model.T, domain=pe.NonNegativeReals)	    #Power out of battery at time t
            self.model.wbc_csp = pe.Var(self.model.T, domain=pe.NonNegativeReals)	    #Power into battery at time t
            self.model.wbc_pv = pe.Var(self.model.T, domain=pe.NonNegativeReals)	    #Power from PV directly charging the battery at time t
            self.model.wpv = pe.Var(self.model.T, domain=pe.NonNegativeReals)    #Power from PV at time t
            
            self.model.i_p = pe.Var(self.model.T, domain=pe.NonNegativeReals)	    #Battery current for charge in time t
            self.model.i_n = pe.Var(self.model.T, domain=pe.NonNegativeReals)	    #Battery current for discharge in time t
            
            self.model.x_p = pe.Var(self.model.T, domain=pe.NonNegativeReals)	    #Aux Var, B/C product at time t
            self.model.x_n = pe.Var(self.model.T, domain=pe.NonNegativeReals)	    #Aux Var, B/C product at time t
            self.model.z_p = pe.Var(self.model.T, domain=pe.NonNegativeReals)	    #Aux Var, C/C product at time t
            self.model.z_n = pe.Var(self.model.T, domain=pe.NonNegativeReals)	    #Aux Var, C/C product at time t
            
            self.model.bat_lc  = pe.Var(domain=pe.NonNegativeReals)
        
        #--------------- Binary Variables ----------------------
        self.model.yr = pe.Var(self.model.T, domain=pe.Binary)        #1 if receiver is generating ``usable'' thermal power at period $t$; 0 otherwise
        self.model.yrhsp = pe.Var(self.model.T, domain=pe.Binary)	    #1 if receiver hot start-up penalty is incurred at period $t$ (from standby); 0 otherwise
        self.model.yrsb = pe.Var(self.model.T, domain=pe.Binary)	    #1 if receiver is in standby mode at period $t$; 0 otherwise
        self.model.yrsd = pe.Var(self.model.T, domain=pe.Binary)	    #1 if receiver is shut down at period $t$; 0 otherwise
        self.model.yrsu = pe.Var(self.model.T, domain=pe.Binary)      #1 if receiver is starting up at period $t$; 0 otherwise
        self.model.yrsup = pe.Var(self.model.T, domain=pe.Binary)     #1 if receiver cold start-up penalty is incurred at period $t$ (from off); 0 otherwise
        self.model.y = pe.Var(self.model.T, domain=pe.Binary)         #1 if cycle is generating electric power at period $t$; 0 otherwise
        self.model.ychsp = pe.Var(self.model.T, domain=pe.Binary)     #1 if cycle hot start-up penalty is incurred at period $t$ (from standby); 0 otherwise
        self.model.ycsb = pe.Var(self.model.T, domain=pe.Binary)      #1 if cycle is in standby mode at period $t$; 0 otherwise
        self.model.ycsd = pe.Var(self.model.T, domain=pe.Binary)	    #1 if cycle is shutting down at period $t$; 0 otherwise
        self.model.ycsu = pe.Var(self.model.T, domain=pe.Binary)      #1 if cycle is starting up at period $t$; 0 otherwise
        self.model.ycsup = pe.Var(self.model.T, domain=pe.Binary)     #1 if cycle cold start-up penalty is incurred at period $t$ (from off); 0 otherwise
        self.model.ycgb = pe.Var(self.model.T, domain=pe.NonNegativeReals)      #1 if cycle begins electric power generation at period $t$; 0 otherwise
        self.model.ycge = pe.Var(self.model.T, domain=pe.NonNegativeReals)      #1 if cycle stops electric power generation at period $t$; 0 otherwise
        
        #--------------- Persistence Variables ----------------------
        if self.include["persistence"]:
            self.model.wdot_s_prev_delta_plus = pe.Var(self.model.T, domain=pe.NonNegativeReals)
            self.model.wdot_s_prev_delta_minus = pe.Var(self.model.T, domain=pe.NonNegativeReals)           
            self.model.ycoff = pe.Var(self.model.T, domain=pe.Binary)     #1 if power cycle is off at period $t$; 0 otherwise
        
        #----------Binary Battery Variables---------------------
        if self.include["battery"]:
            self.model.ybc = pe.Var(self.model.T, domain=pe.Binary)    #1 if charging battery in t, 0 o.w.
            self.model.ybd = pe.Var(self.model.T, domain=pe.Binary)    #1 if discharging battery in t, 0 o.w.
        
        #----------Binary PV Variables---------------------
        if self.include["pv"]:
            self.model.ypv = pe.Var(self.model.T, domain=pe.Binary)    #1 if PV is feeding the AC system in t, 0 o.w.
           
                
    def addObjective(self):
        def objectiveRule(model):
            return (
                    sum( model.D[t] * 
                    #obj_profit
                    model.Delta[t]*model.P[t]*0.1*(model.wdot_s[t] - model.wdot_p[t])
                    #obj_cost_cycle_su_hs_sd
                    - (model.Ccsu*model.ycsup[t] + 0.1*model.Cchsp*model.ychsp[t] + model.alpha*model.ycsd[t])
                    #obj_cost_cycle_ramping
                    - (model.C_delta_w*(model.wdot_delta_plus[t]+model.wdot_delta_minus[t])+model.C_v_w*(model.wdot_v_plus[t] + model.wdot_v_minus[t]))
                    #obj_cost_rec_su_hs_sd
                    - (model.Crsu*model.yrsup[t] + model.Crhsp*model.yrhsp[t] + model.alpha*model.yrsd[t])
                    #obj_cost_ops
                    - model.Delta[t]*(model.Cpc*model.wdot[t] + model.Ccsb*model.Qb*model.ycsb[t] + model.Crec*model.xr[t] )
                    for t in model.T) 
                    )
        self.model.OBJ = pe.Objective(rule=objectiveRule, sense = pe.maximize)
        
    def addPersistenceConstraints(self):
        def wdot_s_persist_pos_rule(model,t):
            return model.wdot_s_prev_delta_plus[t] >= model.wdot_s[t] - model.wdot_s_prev[t]
        def wdot_s_persist_neg_rule(model,t):
            return model.wdot_s_prev_delta_minus[t] >= model.wdot_s_prev[t] - model.wdot_s[t]
        self.model.persist_pos_con = pe.Constraint(self.model.T,rule=wdot_s_persist_pos_rule)
        self.model.persist_neg_con = pe.Constraint(self.model.T,rule=wdot_s_persist_neg_rule)
            
    def generateConstraints(self):
        if self.include["persistence"]:
            self.addPersistenceConstraints()
        self.addReceiverStartupConstraints()
    
    def addReceiverStartupConstraints(self):
        def rec_inventory_rule(model,t):
            if t == 1:
                return model.ursu[t] <= model.ursu0 + model.Delta[t]*model.xrsu[t]
            return model.ursu[t] <= model.ursu[t-1] + model.Delta[t]*model.xrsu[t]
        def rec_inv_nonzero_rule(model,t):
            return model.ursu[t] <= model.Er * model.yrsu[t]
        def rec_startup_rule(model,t):
            if t == 1:
                return model.yr[t] <= model.ursu[t]/model.Er + model.yr0 + model.yrsb0
            return model.yr[t] <= model.ursu[t]/model.Er + model.yr[t-1] + model.yrsb[t-1]
        def rec_su_persist_rule(model,t):
            if t == 1: 
                return model.yrsu[t] + model.yr0 <= 1
            return model.yrsu[t] +  model.yr[t-1] <= 1
        def ramp_limit_rule(model,t):
            return model.xrsu[t] <= model.Qru*model.yrsu[t]
        def nontrivial_solar_rule(model,t):
            return model.yrsu[t] <= model.Qin[t]
        self.model.rec_inventory_con = pe.Constraint(self.model.T,rule=rec_inventory_rule)
        self.model.rec_inv_nonzero_con = pe.Constraint(self.model.T,rule=rec_inv_nonzero_rule)
        self.model.rec_startup_con = pe.Constraint(self.model.T,rule=rec_startup_rule)
        self.model.rec_su_persist_con = pe.Constraint(self.model.T,rule=rec_su_persist_rule)
        self.model.ramp_limit_con = pe.Constraint(self.model.T,rule=ramp_limit_rule)
        self.model.nontrivial_solar_con = pe.Constraint(self.model.T,rule=nontrivial_solar_rule)
    
if __name__ == "__main__": 
    params = {"num_periods":24} 
    include = {"pv":False,"battery":False,"persistence":True}
    rt = RealTimeDispatchModel(params,include)
#    rt.model.OBJ.pprint()
    rt.model.ramp_limit_con.pprint()
    
# -*- coding: utf-8 -*-
"""
Pyomo real-time dispatch model

"""
#import pyomo
import pyomo.environ as pe
from pyomo.environ import units
units.load_definitions_from_strings(['USD = [currency]'])
from pyomo.util.check_units import assert_units_consistent, assert_units_equivalent
#TODO either update specifications for temperatures throughout, or convert upon initialization of dispatch model

# We need to modify signal handling when Pyomo is called from inside the
# webserver. John Siirola says to use these two lines here, so I guess this is
# gospel: https://github.com/PyUtilib/pyutilib/issues/31#issuecomment-382479024
import pyutilib.subprocess.GlobalData
pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False

class RealTimeDispatchModel(object):
    def __init__(self, params, include={"pv": False, "battery": False, "persistence": False}):
        """
        Generates a concrete instance of a pyomo model for CSP dispatch optimization.
        :param params: DispatchParams object containing inputs, taken from SSC or converted from files
        :param include: settings of optional add-ons, such as battery, PV and operational assumptions
        """
        self.model = pe.ConcreteModel()
        self.include = include
        self.generateParams(params)
        self.generateVariables()
        self.addObjective()
        self.generateConstraints()
        assert_units_equivalent(self.model.OBJ, units.USD)
        assert_units_consistent(self.model)

    def generateParams(self,params):
        """

        :param params: DispatchParams object containing inputs, taken from SSC or converted from files
        :return:
        """
        ### Sets and Indices ###
        # Time indexing
        self.model.num_periods = pe.Param(initialize=params.T)                 #Number of time steps in time-indexed params
        self.model.t_start = pe.Param(initialize=params.start)            #First index of the problem
        self.model.t_end = pe.Param(initialize=params.stop)               #Last index of the problem
        self.model.t_transition = pe.Param(initialize=params.transition)  #Index of the transition between models (last index of non-linear formulation)
        self.model.T = pe.Set(initialize=range(params.start, params.stop+1))  #Set of time steps in the problem
        self.model.T_nl = pe.Set(initialize=range(params.start, params.transition+1))  #Set of periods of the non-linear formulation
        self.model.T_l = pe.Set(initialize=range(params.transition+1, params.stop+1))  #Set of periods of the linear formulation
        self.model.T_inputs = pe.Set(initialize=range(1, params.T+1))
        # Hours for signal
        if self.include["signal"]:
            self.model.num_signal_hours = pe.Param(initialize=params.num_signal_hours)
            self.model.H = pe.Set(initialize=range(1, params.num_signal_hours+1))
            def T_h_init(model, h):
                for t in model.T:
                    if (h-1)+1e-10 < params.Delta_e[t] <= h:
                        yield t
            self.model.T_h = pe.Set(self.model.H, initialize=T_h_init)

        # Piecewise-linear indexing
        self.model.nc = pe.Param(initialize=params.nc)     # Number of linear regions in htf pumping power through the cycle regression [-]
        self.model.htf_segments = pe.Set(initialize=range(1, params.nc+1))
        self.model.nfw = pe.Param(initialize=params.nfw)   # Number of linear regions in feed water pumping power through the cycle regression [-]
        self.model.fw_segments = pe.Set(initialize=range(1, params.nfw + 1))

        #------- Piecewise-linear indexed parameters ------------
        self.model.Pc = pe.Param(self.model.htf_segments, initialize=params.Pc, units=(units.kW*units.s)/units.kg)  #Slope of the htf pumping power through the cycle regression region i [kWe / kg/s]
        self.model.Bc = pe.Param(self.model.htf_segments, initialize=params.Bc, units=units.kW)  #Intercept of the htf pumping power through the cycle regression region i [kWe]
        self.model.Pfw = pe.Param(self.model.fw_segments, initialize=params.Pfw, units=(units.kW*units.s)/units.kg)  #Slope of the feed water pumping power through the cycle regression region i [kWe / kg/s]
        self.model.Bfw = pe.Param(self.model.fw_segments, initialize=params.Bfw, units=units.kW)  #Intercept of the feed water pumping power through the cycle regression region i [kWe]

        #------- Time-indexed parameters --------------
        self.model.Delta = pe.Param(self.model.T_inputs, mutable=False, initialize=params.Delta, units=units.hr)          #duration of period t
        self.model.Delta_e = pe.Param(self.model.T_inputs, mutable=False, initialize=params.Delta_e, units=units.hr)       #cumulative time elapsed at end of period t
        ### Time-series CSP Parameters ##
        # self.model.delta_rs = pe.Param(self.model.T_inputs, mutable=True, initialize=params.delta_rs) # \delta^{rs}_{t}: Estimated fraction of period $t$ required for receiver start-up [-]
        # self.model.delta_cs = pe.Param(self.model.T_inputs, mutable=True, initialize=params.delta_cs)  # \delta^{cs}_{t}: Estimated fraction of period $t$ required for cycle start-up [-]
        self.model.D = pe.Param(self.model.T_inputs, mutable=True, initialize=params.D) #D_{t}: Time-weighted discount factor in period $t$ [-]
        self.model.etaamb = pe.Param(self.model.T_inputs, mutable=True, initialize=params.etaamb)  #\eta^{amb}_{t}: Cycle efficiency ambient temperature adjustment factor in period $t$ [-]
        self.model.etac = pe.Param(self.model.T_inputs, mutable=True, initialize=params.etac)   #\eta^{c}_{t}:   Normalized condenser parasitic loss in period $t$ [-]
        self.model.F = pe.Param(self.model.T_inputs, mutable=True, initialize=params.F)   ##F_{t}: Ratio of actual to clear-sky CSP heliostat field power in period $t$ [-]
        self.model.P = pe.Param(self.model.T_inputs, mutable=True, initialize=params.P, units=units.USD/units.kWh)       #P_{t}: Electricity sales price in period $t$ [\$/kWh\sse]
        self.model.Qin = pe.Param(self.model.T_inputs, mutable=True, initialize=params.Qin, units=units.kW)    #Q^{in}_{t}: Available thermal power generated by the CSP heliostat field in period $t$ [kW\sst]
        # self.model.Qc = pe.Param(self.model.T_inputs, mutable=True, initialize=params.Qc, units=units.kW)     #Q^{c}_{t}: Allowable power per period for cycle start-up in period $t$ [kW\sst]
        self.model.Wdotnet = pe.Param(self.model.T_inputs, mutable=True, initialize=params.Wdotnet, units=units.kW)  #\dot{W}^{net}_{t}: Net grid transmission upper limit in period $t$ [kW\sse]
        self.model.W_u_plus = pe.Param(self.model.T_inputs, mutable=True, initialize=params.W_u_plus, units=units.kW)  #W^{u+}_{t}: Maximum power production when starting generation in period $t$  [kW\sse]
        self.model.W_u_minus = pe.Param(self.model.T_inputs, mutable=True, initialize=params.W_u_minus, units=units.kW)  #W^{u-}_{t}: Maximum power production in period $t$ when stopping generation in period $t+1$  [kW\sse]
        Q_cls_d = dict()
        for t in self.model.T_inputs:
            Q_cls_d[t] = self.model.Qin[t] / (1 if self.model.F[t].value == 0 else self.model.F[t])
        self.model.Q_cls = pe.Param(self.model.T_inputs, mutable=True, initialize=Q_cls_d, units=units.kW)  # Calculated theoretic clear-sky resource at period t [kW\sst]

        ### Time-Series PV Parameters ###
        if self.include["pv"]:
            self.model.wpv_dc = pe.Param(self.model.T_inputs, mutable=True, initialize=params.wpv_dc, units=units.kW)      #w^{PV}_t: maximum DC power production from PV system in period $t$
        
        ###  Cost Parameters ###
        self.model.alpha = pe.Param(mutable=True, initialize=params.alpha, units=units.USD)        #alpha: Conversion factor between unitless and monetary values [\$]
        self.model.Crec = pe.Param(mutable=True, initialize=params.Crec, units=units.USD/units.kWh)         #Crec: Operating cost of heliostat field and receiver [\$/kWh\sst]
        self.model.Crsu = pe.Param(mutable=True, initialize=params.Crsu, units=units.USD)         #Crsu: Penalty for receiver cold start-up [\$/start]
        self.model.Crhsp = pe.Param(mutable=True, initialize=params.Crhsp, units=units.USD)        #Crhsp: Penalty for receiver hot start-up [\$/start]
        self.model.Cpc = pe.Param(mutable=True, initialize=params.Cpc, units=units.USD/units.kWh)          #Cpc: Operating cost of power cycle [\$/kWh\sse]
        self.model.Ccsu = pe.Param(mutable=True, initialize=params.Ccsu, units=units.USD)        #Ccsu: Penalty for power cycle cold start-up [\$/start]
        self.model.Cchsp = pe.Param(mutable=True, initialize=params.Cchsp, units=units.USD)       #Cchsp: Penalty for power cycle hot start-up [\$/start]
        self.model.C_delta_w = pe.Param(mutable=True, initialize=params.C_delta_w, units=units.USD/units.kW)    #C_delta_w: Penalty for change in power cycle  production [\$/$\Delta\text{kW}$\sse]
        self.model.C_v_w = pe.Param(mutable=True, initialize=params.C_v_w, units=units.USD/units.kW)        #C_v_w: Penalty for change in power cycle  production \tcb{beyond designed limits} [\$/$\Delta\text{kW}$\sse]
        self.model.Ccsb = pe.Param(mutable=True, initialize=params.Ccsb, units=units.USD/units.hr)         #Ccsb: Operating cost of power cycle standby operation [\$/h]
        if self.include["signal"]:
            self.model.Cg_plus = pe.Param(self.model.H, mutable=True, initialize=params.Cg_plus, units=units.USD/units.kWh)         #Cg_plus: Penalty for overproducing in period  [\$/kWh\sst]
            self.model.Cg_minus = pe.Param(self.model.H, mutable=True, initialize=params.Cg_minus, units=units.USD/units.kWh)  # Cg_minus: Penalty for overproducing in period  [\$/kWh\sst]
            self.model.day_ahead_tol_plus = pe.Param(mutable=True, initialize=params.day_ahead_tol_plus, units=units.kWh)  # Cg_minus: Tolerance for overproducing in period  [kWh\sst]
            self.model.day_ahead_tol_minus = pe.Param(mutable=True, initialize=params.day_ahead_tol_minus, units=units.kWh)  # Cg_minus: Tolerance for underproducing in period  [kWh\sst]
        
        # -------PV and Battery Cost Parameters -------
        if self.include["pv"]:
            self.model.Cpv = pe.Param(mutable=True, initialize=params.Cpv, units=units.USD/units.kWh)    #Operating cost of photovoltaic field [\$/kWh\sse]
        if self.include["battery"]:
            self.model.Cbc = pe.Param(mutable=True, initialize=params.Cbc, units=units.USD/units.kWh)    #Operating cost of charging battery [\$/kWh\sse]
            self.model.Cbd = pe.Param(mutable=True, initialize=params.Cbd, units=units.USD/units.kWh)    #Operating cost of discharging battery [\$/kWh\sse]
            self.model.Cbl = pe.Param(mutable=True, initialize=params.Cbl, units=units.USD)    #Lifecycle cost for battery [\$/lifecycle]
        
        ### CSP Field and Receiver Parameters ###
        if self.include["simple_receiver"]:
            self.model.P_field_rec = pe.Param(self.model.T_inputs, mutable=True, initialize=params.P_field_rec, units=units.kW)  #Parasitic power draws from receiver operations [kW\sse]
        else:
            #self.model.deltal = pe.Param(mutable=True, initialize=params.deltal, units=units.hr)  # Minimum time to start the receiver [hr]
            self.model.Drsu = pe.Param(mutable=True, initialize=params.Drsu, units=units.hr)    #Minimum time to start the receiver [hr]
            self.model.Drsd = pe.Param(mutable=True, initialize=params.Drsd, units=units.hr)    #Minimum time to shut down the receiver [hr]
            self.model.Ehs = pe.Param(mutable=True, initialize=params.Ehs, units=units.kWh)       #Heliostat field startup or shut down parasitic loss [kWh\sse]
            self.model.Er = pe.Param(mutable=True, initialize=params.Er, units=units.kWh)        #Required energy expended to start receiver [kWh\sst]
            self.model.Lr = pe.Param(mutable=True, initialize=params.Lr)        #Receiver pumping power per unit power produced [kW\sse/kW\sst]
            self.model.mdot_r_min = pe.Param(mutable=True, initialize=params.mdot_r_min, units=units.kg/units.s)  # Minimum mass flow rate of heat transfer fluid to the receiver [kg/s]
            self.model.mdot_r_max = pe.Param(mutable=True, initialize=params.mdot_r_max, units=units.kg/units.s)  # Maximum mass flow rate of heat transfer fluid to the receiver [kg/s]
            self.model.Pr = pe.Param(mutable=True, initialize=params.Pr, units=(units.kW*units.s)/units.kg)  # Receiver pumping power per unit mass flow rate [kW\sse/ kg/s]
            self.model.Qrl = pe.Param(mutable=True, initialize=params.Qrl, units=units.kW)       #Minimum operational thermal power delivered by receiver [kWh\sst]
            self.model.Qrsb = pe.Param(mutable=True, initialize=params.Qrsb, units=units.kW)      #Required thermal power for receiver standby [kWh\sst]
            self.model.Qrsd = pe.Param(mutable=True, initialize=params.Qrsd, units=units.kWh)      #Required thermal power for receiver shut down [kWh\sst]
            self.model.Qru = pe.Param(mutable=True, initialize=params.Qru, units=units.kW)       #Allowable power per period for receiver start-up [kWh\sst]
            # self.model.T_rout_min = pe.Param(mutable=True, initialize=params.T_rout_min)  # Minimum allowable receiver outlet temperature [deg C]
            self.model.T_rout_max = pe.Param(mutable=True, initialize=params.T_rout_max, units=units.degK)  # Maximum allowable receiver outlet temperature [deg C]
            self.model.Wh_comm = pe.Param(mutable=True, initialize=params.Wh_comm, units=units.kW)  # Heliostat field communication parasitic loss [kW\sse]
            self.model.Wh_track = pe.Param(mutable=True, initialize=params.Wh_track, units=units.kW)  # Heliostat field tracking parasitic loss [kW\sse]
            self.model.Wht_full = pe.Param(mutable=True, initialize=params.Wht_full, units=units.kW)  # Tower piping heat trace full load parasitic loss [kW\sse]
            self.model.Wht_part = pe.Param(mutable=True, initialize=params.Wht_part, units=units.kW)  # Tower piping heat trace part load parasitic loss [kW\sse]
            #self.model.Wh = pe.Param(mutable=True, initialize=params.Wh, units=units.kW)        #Heliostat field tracking parasitic loss [kW\sse]
            #self.model.Wht = pe.Param(mutable=True, initialize=params.Wht, units=units.kW)       #Tower piping heat trace parasitic loss [kW\sse]

        ### Thermal Energy Storage parameters
        self.model.Eu = pe.Param(mutable=True, units=units.kWh, initialize=params.Eu)  # Thermal energy storage capacity [kWh\sst]
        self.model.Cp = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.Cp, units=units.kJ/(units.degK*units.kg))  # Specific heat of the heat transfer fluid & [kJ\sst/ kg $^{\circ} C$]
        self.model.mass_cs_min = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.mass_cs_min, units=units.kg)  # Minimum mass of heat transfer fluid in cold storage [kg]
        self.model.mass_cs_max = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.mass_cs_max, units=units.kg)  # Maximum mass of heat transfer fluid in cold storage [kg]
        self.model.mass_hs_min = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.mass_hs_min, units=units.kg)  # Minimum mass of heat transfer fluid in hot storage [kg]
        self.model.mass_hs_max = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.mass_hs_max, units=units.kg)  # Maximum mass of heat transfer fluid in hot storage [kg]
        self.model.T_cs_min = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.T_cs_min, units=units.degK)  # Minimum temperature of heat transfer fluid in cold storage [C]
        self.model.T_cs_max = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.T_cs_max, units=units.degK)  # Maximum temperature of heat transfer fluid in cold storage [C]
        self.model.T_hs_min = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.T_hs_min, units=units.degK)  # Minimum temperature of heat transfer fluid in hot storage [C]
        self.model.T_hs_max = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.T_hs_max, units=units.degK)  # Maximum temperature of heat transfer fluid in hot storage [C]
        self.model.T_cs_des = pe.Param(mutable=True, within=pe.NonNegativeReals,
                                       initialize=params.T_cs_des, units=units.degK)  # Design point temperature of heat transfer fluid in cold storage [C]
        self.model.T_hs_des = pe.Param(mutable=True, within=pe.NonNegativeReals,
                                       initialize=params.T_hs_des, units=units.degK)  # Design point temperature of heat transfer fluid in hot storage [C]

        ### Power Cycle Parameters ###
        self.model.alpha_b = pe.Param(mutable=True, within=pe.Reals, initialize=params.alpha_b*params.delta_T_design, units=units.degK)  #Regression coefficients for heat transfer fluid temperature drop across SGS model
        self.model.alpha_T = pe.Param(mutable=True, within=pe.Reals, initialize=params.alpha_T*params.delta_T_design/params.T_cin_design)  #Regression coefficients for heat transfer fluid temperature drop across SGS model
        self.model.alpha_m = pe.Param(mutable=True, within=pe.Reals, initialize=params.alpha_m*params.delta_T_design/params.mdot_c_design, units=units.degK*units.s/units.kg)  #Regression coefficients for heat transfer fluid temperature drop across SGS model
        self.model.beta_b = pe.Param(mutable=True, within=pe.Reals, initialize=params.beta_b*params.Wdot_design, units=units.kW)    #Regression coefficients for the power cycle efficiency model
        self.model.beta_T = pe.Param(mutable=True, within=pe.Reals, initialize=params.beta_T*params.Wdot_design/params.T_cin_design, units=units.kW/units.degK)    #Regression coefficients for the power cycle efficiency model
        self.model.beta_m = pe.Param(mutable=True, within=pe.Reals, initialize=params.beta_m*params.Wdot_design/params.mdot_c_design, units=units.kW*units.s/units.kg)    #Regression coefficients for the power cycle efficiency model
        self.model.beta_mT = pe.Param(mutable=True, within=pe.Reals, initialize=params.beta_mT*params.Wdot_design/(params.mdot_c_design*params.T_cin_design), units=units.kW*units.s/(units.kg*units.degK))  #Regression coefficients for the power cycle efficiency model
        self.model.delta_T_design = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.delta_T_design, units=units.degK)  #Design point temperature change of the heat transfer fluid across the SGS model
        self.model.delta_T_max = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.delta_T_max, units=units.degK)   #Max temperature change of the heat transfer fluid across the SGS model
        self.model.Ec = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.Ec, units=units.kWh)           #Required energy expended to cold start cycle [kWh\sst]
        self.model.Ew = pe.Param(mutable=True, within=pe.NonNegativeReals,
                                 initialize=params.Ew, units=units.kWh)  # Required energy expended to warm start cycle (from standby) [kWh\sst]
        self.model.eta_des = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.eta_des)      #Cycle nominal efficiency [-]
        self.model.etap = pe.Param(mutable=True, within=pe.Reals, initialize=params.etap)         #Slope of linear approximation of power cycle performance curve [kW\sse/kW\sst]
        self.model.kl = pe.Param(mutable=True, within=pe.Reals, initialize=params.kl, units=units.kW/units.degK)     #Change in lower bound of cycle thermal load due to hot storage temperature
        self.model.ku = pe.Param(mutable=True, within=pe.Reals, initialize=params.ku, units=units.kW/units.degK)     #Change in upper bound of cycle thermal load due to hot storage temperature
        self.model.Lc = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.Lc)           #Cycle heat transfer fluid pumping power per unit energy expended [kW\sse/kW\sst]
        self.model.mdot_c_design = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.mdot_c_design, units=units.kg/units.s)  #Design point mass flow rate of the heat transfer fluid through the power cycle
        self.model.mdot_c_min = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.mdot_c_min, units=units.kg/units.s)  #Minimum mass flow rate of heat transfer fluid to the cycle [kg/s]
        self.model.mdot_c_max = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.mdot_c_max, units=units.kg/units.s)  #Maximum mass flow rate of heat transfer fluid to the cycle [kg/s]
        self.model.Qc = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.Qc, units=units.kW)  # Thermal power input to power cycle during start-up [kW\sst]
        # self.model.Qb = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.Qb, units=units.kW)           #Cycle standby thermal power consumption per period [kW\sst]
        self.model.Ql = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.Ql, units=units.kW)           #Minimum operational thermal power input to cycle [kW\sst]
        self.model.Qu = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.Qu, units=units.kW)           #Cycle thermal power capacity [kW\sst]
        self.model.T_cin_design = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.T_cin_design, units=units.degK)   #Design point power cycle inlet temperature of the heat transfer fluid
        # self.model.T_cout_min = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.T_cout_min, units=units.degK)   #Minimum allowable cycle outlet temperature [deg C]
        # self.model.T_cout_max = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.T_cout_max, units=units.degK)  #Maximum allowable cycle outlet temperature [deg C]
        self.model.Wdot_design = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.Wdot_design, units=units.kW) #Design point electrical output of the power cycle
        self.model.Wdot_p_max = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.Wdot_p_max, units=units.kW) #Power purchase required to cover all parasitic loads [kW\sse]
        self.model.Wb = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.Wb, units=units.kW)           #Power cycle standby operation parasitic load [kW\sse]
        self.model.Wc = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.Wc, units=units.kW)           #Power cycle operating parasitic loss [kW\sse]
        self.model.Wdotl = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.Wdotl, units=units.kW)        #Minimum cycle electric power output [kW\sse]
        self.model.Wdotu = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.Wdotu, units=units.kW)        #Cycle electric power rated capacity [kW\sse]
        self.model.W_delta_plus = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.W_delta_plus, units=units.kW/units.hr) #Power cycle ramp-up designed limit [kW\sse/h]
        self.model.W_delta_minus = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.W_delta_minus, units=units.kW/units.hr)#Power cycle ramp-down designed limit [kW\sse/h]
        self.model.W_v_plus = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.W_v_plus, units=units.kW)     #Power cycle ramp-up violation limit [kW\sse/h]
        self.model.W_v_minus = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.W_v_minus, units=units.kW)    #Power cycle ramp-down violation limit [kW\sse/h]
        self.model.Yu = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.Yu, units=units.hr)           #Minimum required power cycle uptime [h]
        self.model.Yd = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.Yd, units=units.hr)           #Minimum required power cycle downtime [h]
        
        ### Initial Condition Parameters ###
        self.model.drsd0 = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.drsd0, units=units.hr)  #Time spent shutting down the receiver before the problem horizon [h]
        self.model.drsu0 = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.drsu0, units=units.hr)  #Time spent starting up the receiver before the problem horizon [h]
        # self.model.s0 = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.s0, units=units.kWh)  #Initial TES reserve quantity  [kWh\sst]  -- moved to transition
        self.model.mass_cs0 = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.mass_cs0, units=units.kg)  #Initial mass of heat transfer fluid in cold storage  [kg]
        self.model.mass_hs0 = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.mass_hs0, units=units.kg)  #Initial mass of heat transfer fluid in hot storage  [kg]
        self.model.T_cs0 = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.T_cs0, units=units.degK)  #Initial temperature of heat transfer fluid in cold storage  [C]
        self.model.T_hs0 = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.T_hs0, units=units.degK)  #Initial temperature of heat transfer fluid in hot storage  [C]
        self.model.ucsu0 = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.ucsu0, units=units.kWh) #Initial cycle start-up energy inventory  [kWh\sst]
        self.model.ursd0 = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.ursd0, units=units.kWh) #Initial receiver shut-down energy inventory [kWh\sst]
        self.model.ursu0 = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.ursu0, units=units.kWh)  # Initial receiver start-up energy inventory [kWh\sst]
        self.model.wdot0 = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.wdot0, units=units.kW) #Initial power cycle electricity generation [kW\sse]
        self.model.yr0 = pe.Param(mutable=True, within=pe.NonNegativeIntegers, initialize=params.yr0)  #1 if receiver is generating ``usable'' thermal power initially, 0 otherwise  [az] this is new.
        self.model.yrsb0 = pe.Param(mutable=True, within=pe.NonNegativeIntegers, initialize=params.yrsb0)  #1 if receiver is in standby mode initially, 0 otherwise [az] this is new.
        self.model.yrsu0 = pe.Param(mutable=True, within=pe.NonNegativeIntegers, initialize=params.yrsu0)  #1 if receiver is in starting up initially, 0 otherwise    [az] this is new.
        self.model.yrsd0 = pe.Param(mutable=True, within=pe.NonNegativeIntegers, initialize=params.yrsd0)  #1 if receiver is in shutting down initially, 0 otherwise    [az] this is new.
        self.model.y0 = pe.Param(mutable=True, within=pe.NonNegativeIntegers, initialize=params.y0)  #1 if cycle is generating electric power initially, 0 otherwise
        self.model.ycsb0 = pe.Param(mutable=True, within=pe.NonNegativeIntegers, initialize=params.ycsb0)  #1 if cycle is in standby mode initially, 0 otherwise
        self.model.ycsu0 = pe.Param(mutable=True, within=pe.NonNegativeIntegers, initialize=params.ycsu0)  #1 if cycle is in starting up initially, 0 otherwise    [az] this is new.
        self.model.Yu0 = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.Yu0)  # duration that cycle has been generating electric power [h]
        self.model.Yd0 = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.Yd0)  # duration that cycle has not been generating power (i.e., shut down or in standby mode) [h]
        # -------Persistence Parameters ---------
        if self.include["persistence"]:
            self.model.wdot_s_prev  = pe.Param(self.model.T_inputs, within=pe.NonNegativeReals, mutable=True, initialize=params.wdot_s_prev)
            self.model.wdot_s_pen  = pe.Param(self.model.T_inputs, within=pe.Reals, mutable=True, initialize=params.wdot_s_pen)
        
        # -------Miscellaneous Parameters taken from SAM---------
        # self.model.day_of_year = pe.Param(mutable=True, initialize=params.day_of_year)
        self.model.disp_time_weighting = pe.Param(mutable=True, within=pe.Reals, initialize=params.disp_time_weighting)
        # self.model.csu_cost = pe.Param(mutable=True, initialize=params.csu_cost)
        # self.model.eta_cycle = pe.Param(mutable=True, initialize=params.eta_cycle)
        # self.model.gamma = pe.Param(mutable=True, initialize=params.gamma)
        # self.model.gammac = pe.Param(mutable=True, initialize=params.gammac)
        # self.model.M = pe.Param(mutable=True, initialize=params.M)
        # self.model.qrecmaxobs = pe.Param(mutable=True, initialize=params.qrecmaxobs)
        # self.model.W_dot_cycle = pe.Param(mutable=True, initialize=params.W_dot_cycle)
        # self.model.Z_1 = pe.Param(mutable=True, initialize=params.Z_1)
        # self.model.Z_2 = pe.Param(mutable=True, initialize=params.Z_2)
        # self.model.max_up = pe.Param(mutable=True, initialize=params.max_up)
        # self.model.max_down = pe.Param(mutable=True, initialize=params.max_down)
        # self.model.max_up_v = pe.Param(mutable=True, initialize=params.max_up_v)
        # self.model.max_down_v = pe.Param(mutable=True, initialize=params.max_down_v)
        # self.model.pen_delta_w = pe.Param(mutable=True, initialize=params.pen_delta_w)
        # self.model.q0 = pe.Param(mutable=True, initialize=params.q0)
        # self.model.rsu_cost = pe.Param(mutable=True, initialize=params.rsu_cost)
        # self.model.tdown0 = pe.Param(mutable=True, initialize=params.tdown0)
        # self.model.tstby0 = pe.Param(mutable=True, initialize=params.tstby0)
        # self.model.tup0 = pe.Param(mutable=True, initialize=params.tup0)
        # self.model.Wdot0 = pe.Param(mutable=True, initialize=params.Wdot0)
        # self.model.wnet_lim_min = pe.Param(self.model.T_inputs, mutable=True, initialize=params.wnet_lim_min)
        # self.model.cap_frac = pe.Param(self.model.T_inputs, mutable=True, initialize=params.cap_frac)
        # self.model.eff_frac = pe.Param(self.model.T_inputs, mutable=True, initialize=params.eff_frac)
        # self.model.dt = pe.Param(self.model.T_inputs, mutable=True, initialize=params.dt)
        # self.model.dte = pe.Param(self.model.T_inputs, mutable=True, initialize=params.dte)
        # self.model.twt = pe.Param(self.model.T_inputs, mutable=True, initialize=params.twt)
        self.model.avg_price = pe.Param(mutable=True, within=pe.NonNegativeReals, units=units.USD/units.kWh, initialize=params.avg_price)  #average sale price [\$/kWh]
        self.model.avg_purchase_price = pe.Param(mutable=True, within=pe.NonNegativeReals, units=units.USD/units.kWh, initialize=params.avg_purchase_price)   #average grid purchase price [\$/kWh]
        self.model.avg_price_disp_storage_incentive = pe.Param(mutable=True, within=pe.NonNegativeReals, units=units.USD/units.kWh, initialize=params.avg_price_disp_storage_incentive )   #average grid purchase price [\$/kWh]

        if self.include["signal"]:
            self.model.G = pe.Param(self.model.H, mutable=True, within=pe.NonNegativeReals, units=units.kWh, initialize=params.G)
        
        
        #--------Parameters for the Battery---------
        if self.include["battery"]:
            self.model.alpha_p = pe.Param(mutable=True, within=pe.Reals, initialize=params.alpha_p)    #Bi-directional converter slope-intercept parameter
            self.model.alpha_n = pe.Param(mutable=True, within=pe.Reals, initialize=params.alpha_n)	  #Bi-directional converter slope-intercept parameter
            self.model.beta_p = pe.Param(mutable=True, within=pe.Reals, initialize=params.beta_p)     #Bi-directional converter slope parameter
            self.model.beta_n = pe.Param(mutable=True, within=pe.Reals, initialize=params.beta_n)	  #Bi-directional converter slope parameter
            self.model.C_B = pe.Param(mutable=True, within=pe.Reals, initialize=params.C_B)
            self.model.I_upper_p = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.I_upper_p)
            self.model.I_upper_n = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.I_upper_n)  #Battery discharge current max
            self.model.S_B_lower = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.S_B_lower)
            self.model.S_B_upper = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.S_B_upper)
            self.model.I_lower_n = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.I_lower_n)
            self.model.I_lower_p = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.I_lower_p)
            self.model.P_B_lower = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.P_B_lower)
            self.model.P_B_upper = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.P_B_upper)  #Battery min/max power rating
            self.model.A_V = pe.Param(mutable=True, within=pe.Reals, initialize=params.A_V)
            self.model.B_V = pe.Param(mutable=True, within=pe.Reals, initialize=params.B_V)	  #Battery linear voltage model slope/intercept coeffs
            self.model.R_int = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.R_int)
            self.model.I_avg = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.I_avg)	  #Typical current expected from the battery
            self.model.alpha_pv = pe.Param(mutable=True, within=pe.Reals, initialize=params.alpha_pv)
            self.model.beta_pv = pe.Param(mutable=True, within=pe.Reals, initialize=params.beta_pv)
            self.model.soc0 = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.soc0)     #initial state of charge
            self.model.Winv_lim = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.Winv_lim)	  # Inverter max power (DC)
            self.model.Wmax = pe.Param(mutable=True, within=pe.NonNegativeReals, initialize=params.Wmax)	  #Constant Max power to grid
            self.model.Winvnt = pe.Param(mutable=True, within=pe.Reals, initialize=params.Winvnt)
            self.model.N_csp = pe.Param(mutable=True, within=pe.Reals, initialize=params.N_csp)
        
        #------------- Cycle Incentive Parameter - for testing only -----------
        if self.include["force_cycle"]:
            self.model.cycle_incent = pe.Param(within=pe.Reals, units=units.USD, initialize=1.0e7)

    def generateVariables(self):
        ### Decision Variables ###
        ##--------- Variables ------------------------
        if not self.include["simple_receiver"]:
            self.model.drsu = pe.Var(self.model.T, domain=pe.NonNegativeReals, units=units.hr)  #Receiver start-up time inventory at period t [h]
            self.model.drsd = pe.Var(self.model.T, domain=pe.NonNegativeReals, units=units.hr)  #Receiver shut down time inventory at period t [h]
            self.model.frsd = pe.Var(self.model.T, domain=pe.NonNegativeReals, bounds = (0,1))  #Fraction of period used for receiver shut down at period $t [-]
            self.model.frsu = pe.Var(self.model.T, domain=pe.NonNegativeReals, bounds = (0,1))  #Fraction of period used for receiver start-up at period $t [-]
            self.model.lr = pe.Var(self.model.T_nl, domain=pe.NonNegativeReals, units=units.kW)  #Salt pumping power to receiver in period t [kW\sse]
        self.model.lc = pe.Var(self.model.T_nl, domain=pe.NonNegativeReals, units=units.kW)  #Salt pumping power to SGS in period t [kW\sse]
        self.model.lfw = pe.Var(self.model.T_nl, domain=pe.NonNegativeReals, units=units.kW)  #Feed water pumping power to SGS in period t [kW\sse]
        self.model.mass_cs = pe.Var(self.model.T_nl, domain=pe.NonNegativeReals, units=units.kg, bounds = (self.model.mass_cs_min,self.model.mass_cs_max))  #Mass of htf in cold storage in period t [kg]
        self.model.mass_hs = pe.Var(self.model.T_nl, domain=pe.NonNegativeReals, units=units.kg, bounds = (self.model.mass_hs_min,self.model.mass_hs_max))  #Mass of htf in hot storage in period t [kg]
        self.model.mdot_c = pe.Var(self.model.T_nl, domain=pe.NonNegativeReals, units=units.kg/units.s)  #Mass flow rate of htf to the cycle in period t [kg/s]
        self.model.mdot_r_cs = pe.Var(self.model.T_nl, domain=pe.NonNegativeReals, units=units.kg/units.s)  #Mass flow rate of htf to the rec to cold in period t [kg/s]
        self.model.mdot_r_hs = pe.Var(self.model.T_nl, domain=pe.NonNegativeReals, units=units.kg/units.s)  #Mass flow rate of htf to the rec to hot in period t [kg/s]
        self.model.s = pe.Var(self.model.T_l, domain=pe.NonNegativeReals, units=units.kWh, bounds = (0,self.model.Eu))                      #TES reserve quantity at period $t$  [kWh\sst]
        self.model.T_cout = pe.Var(self.model.T_nl, domain=pe.NonNegativeReals, units=units.degK)  #Temperature of heat transfer fluid at the cycle outlet in period $t$ & $^{\circ} C$
        self.model.T_cs = pe.Var(self.model.T_nl, domain=pe.NonNegativeReals, units=units.degK, bounds = (self.model.T_cs_min,self.model.T_cs_max))  #Temperature of heat transfer fluid in cold storage in period $t$  & $^{\circ} C$
        self.model.T_hs = pe.Var(self.model.T_nl, domain=pe.NonNegativeReals, units=units.degK, bounds = (self.model.T_hs_min,self.model.T_hs_max))  #Temperature of heat transfer fluid in hot storage in period $t$  & $^{\circ} C$
        self.model.T_rout = pe.Var(self.model.T_nl, domain=pe.NonNegativeReals, units=units.degK)  #Temperature of heat transfer fluid at the receiver outlet in period $t$  & $^{\circ} C$
        self.model.ucsu = pe.Var(self.model.T, domain=pe.NonNegativeReals, units=units.kWh)   #Cycle start-up energy inventory at period $t$ [kWh
        #self.model.ucsd = pe.Var(self.model.T, domain=pe.NonNegativeReals, units=units.kWh)                         #Cycle shutdown energy inventory at period $t$ [kWh\sst]
        self.model.ursu = pe.Var(self.model.T, domain=pe.NonNegativeReals, units=units.kWh)                         #Receiver start-up energy inventory at period $t$ [kWh\sst]
        self.model.ursd = pe.Var(self.model.T,
                                 domain=pe.NonNegativeReals, units=units.kWh)  # Receiver start-up energy inventory at period $t$ [kWh\sst]
        self.model.wdot = pe.Var(self.model.T, domain=pe.NonNegativeReals, units=units.kW)                         #Power cycle electricity generation at period $t$ [kW\sse]
        self.model.wdot_delta_plus = pe.Var(self.model.T, domain=pe.NonNegativeReals, units=units.kW)	             #Power cycle ramp-up in period $t$ [kW\sse]
        self.model.wdot_delta_minus = pe.Var(self.model.T, domain=pe.NonNegativeReals, units=units.kW)	         #Power cycle ramp-down in period $t$ [kW\sse]
        self.model.wdot_v_plus = pe.Var(self.model.T, domain=pe.NonNegativeReals, units=units.kW, bounds = (0,self.model.W_v_plus))      #Power cycle ramp-up beyond designed limit in period $t$ [kW\sse]
        self.model.wdot_v_minus = pe.Var(self.model.T, domain=pe.NonNegativeReals, units=units.kW, bounds = (0,self.model.W_v_minus)) 	 #Power cycle ramp-down beyond designed limit in period $t$ [kW\sse]
        self.model.wdot_s = pe.Var(self.model.T, domain=pe.NonNegativeReals, units=units.kW)	                     #Energy sold to grid in time t
        self.model.wdot_p = pe.Var(self.model.T, domain=pe.NonNegativeReals, units=units.kW)	                     #Energy purchased from the grid in time t
        self.model.x = pe.Var(self.model.T_l, domain=pe.NonNegativeReals, units=units.kW)                            #Cycle thermal power utilization at period $t$ [kW\sst]
        self.model.xr = pe.Var(self.model.T, domain=pe.NonNegativeReals, units=units.kW)	                         #Thermal power delivered by the receiver at period $t$ [kW\sst]
        # if not self.include["simple_receiver"]:    #from Legacy linear-only model
        #     self.model.xrsu = pe.Var(self.model.T, domain=pe.NonNegativeReals, units=units.kW)                         #Receiver start-up power consumption at period $t$ [kW\sst]

        if self.include["signal"]:
            self.model.g_plus = pe.Var(self.model.H,
                                     domain=pe.NonNegativeReals, units=units.kWh)  # overproduction vs. grid signal for demand in hour $h$ [kWh\sse]
            self.model.g_minus = pe.Var(self.model.H,
                                     domain=pe.NonNegativeReals, units=units.kWh)  # underproduction vs. grid signal for demand in hour $h$ [kWh\sse]
        
        #----------Continuous for PV -------------------
        if self.include["pv"]:
            self.model.wpv = pe.Var(self.model.T, domain=pe.NonNegativeReals, units=units.kW)    #Power from PV at time t
        #----------Continuous for the Battery-----------
        if self.include["battery"]:
            self.model.soc = pe.Var(self.model.T, domain=pe.NonNegativeReals, units=units.kWh)	    #State of charge of battery in time t
            self.model.wbd = pe.Var(self.model.T, domain=pe.NonNegativeReals, units=units.kW)	    #Power out of battery at time t
            self.model.wbc_csp = pe.Var(self.model.T, domain=pe.NonNegativeReals, units=units.kW)	    #Power into battery at time t
            if self.include["pv"]:
                self.model.wbc_pv = pe.Var(self.model.T, domain=pe.NonNegativeReals, units=units.kW)	    #Power from PV directly charging the battery at time t
            
            self.model.i_p = pe.Var(self.model.T, domain=pe.NonNegativeReals, units=units.A)	    #Battery current for charge in time t
            self.model.i_n = pe.Var(self.model.T, domain=pe.NonNegativeReals, units=units.A)	    #Battery current for discharge in time t
            
            self.model.x_p = pe.Var(self.model.T, domain=pe.NonNegativeReals)	    #Aux Var, B/C product at time t
            self.model.x_n = pe.Var(self.model.T, domain=pe.NonNegativeReals)	    #Aux Var, B/C product at time t
            self.model.z_p = pe.Var(self.model.T, domain=pe.NonNegativeReals)	    #Aux Var, C/C product at time t
            self.model.z_n = pe.Var(self.model.T, domain=pe.NonNegativeReals)	    #Aux Var, C/C product at time t
            
            self.model.bat_lc  = pe.Var(domain=pe.NonNegativeReals)
        
        #--------------- Binary Variables ----------------------
        if not self.include["simple_receiver"]:
            self.model.yr = pe.Var(self.model.T, domain=pe.Binary)        #1 if receiver is generating ``usable'' thermal power at period $t$; 0 otherwise
            self.model.yrhsp = pe.Var(self.model.T, domain=pe.Binary)	    #1 if receiver hot start-up penalty is incurred at period $t$ (from standby); 0 otherwise
            self.model.yrsb = pe.Var(self.model.T, domain=pe.Binary)	    #1 if receiver is in standby mode at period $t$; 0 otherwise
            self.model.yrsd = pe.Var(self.model.T, domain=pe.Binary)	    #1 if receiver is shut down at period $t$; 0 otherwise
            self.model.yrsdp = pe.Var(self.model.T, domain=pe.Binary)  # 1 if receiver cold shut-down penalty is incurred at period $t$ (from off); 0 oth
            self.model.yrsu = pe.Var(self.model.T, domain=pe.Binary)      #1 if receiver is starting up at period $t$; 0 otherwise
            self.model.yrsup = pe.Var(self.model.T, domain=pe.Binary)     #1 if receiver cold start-up penalty is incurred at period $t$ (from off); 0 otherwise
        self.model.y = pe.Var(self.model.T, domain=pe.Binary)         #1 if cycle is generating electric power at period $t$; 0 otherwise
        self.model.ychsp = pe.Var(self.model.T, domain=pe.Binary)     #1 if cycle hot start-up penalty is incurred at period $t$ (from standby); 0 otherwise
        self.model.ycsb = pe.Var(self.model.T, domain=pe.Binary)      #1 if cycle is in standby mode at period $t$; 0 otherwise
        self.model.ycsd = pe.Var(self.model.T, domain=pe.Binary)	    #1 if cycle is shutting down at period $t$; 0 otherwise
        self.model.ycsdp = pe.Var(self.model.T, domain=pe.Binary)  # 1 if cycle cold shut-down penalty is incurred at time period $t$; 0 otherwise
        self.model.ycsu = pe.Var(self.model.T, domain=pe.Binary)      #1 if cycle is starting up at period $t$; 0 otherwise
        self.model.ycsup = pe.Var(self.model.T, domain=pe.Binary)     #1 if cycle cold start-up penalty is incurred at period $t$ (from off); 0 otherwise
        self.model.ycgb = pe.Var(self.model.T, domain=pe.NonNegativeReals, bounds=(0,1))      #1 if cycle begins electric power generation at period $t$; 0 otherwise
        self.model.ycge = pe.Var(self.model.T, domain=pe.NonNegativeReals, bounds=(0,1))      #1 if cycle stops electric power generation at period $t$; 0 otherwise
        # self.model.ycoff = pe.Var(self.model.T, domain=pe.Binary)  #1 if cycle is not operating in period $t$; 0 otherwise

        #--------------- Persistence Variables ----------------------
        if self.include["persistence"]:
            self.model.wdot_s_prev_delta_plus = pe.Var(self.model.T, domain=pe.NonNegativeReals, units=units.kW)
            self.model.wdot_s_prev_delta_minus = pe.Var(self.model.T, domain=pe.NonNegativeReals, units=units.kW)
            #self.model.ycoff = pe.Var(self.model.T, domain=pe.Binary)     #1 if power cycle is off at period $t$; 0 otherwise
        
        #----------Binary Battery Variables---------------------
        if self.include["battery"]:
            self.model.ybc = pe.Var(self.model.T, domain=pe.Binary)    #1 if charging battery in t, 0 o.w.
            self.model.ybd = pe.Var(self.model.T, domain=pe.Binary)    #1 if discharging battery in t, 0 o.w.
        
        #----------Binary PV Variables---------------------
        if self.include["pv"]:
            self.model.ypv = pe.Var(self.model.T, domain=pe.Binary)    #1 if PV is feeding the AC system in t, 0 o.w.

        #------ Expressions for existing parameters and variables
        self.model.s0 = (
            units.convert(self.model.Cp * (self.model.mass_hs0 - self.model.mass_hs_min) * (
                        self.model.T_hs0 - self.model.T_cs_des), units.kWh )
            if self.model.t_transition == 0 else
            units.convert(self.model.Cp * (self.model.mass_hs[self.model.t_transition] - self.model.mass_hs_min) * (
                    self.model.T_hs[self.model.t_transition] - self.model.T_cs_des), units.kWh )
        )

        self.model.x_calc = lambda t: units.convert(self.model.Cp * self.model.mdot_c[t] * (
                    self.model.T_hs[t] - self.model.T_cout[t]), units.kW)
        self.model.s_calc = lambda t: units.convert(self.model.Cp * (self.model.mass_hs[t] - self.model.mass_hs_min) *
                                                    (self.model.T_hs[t] - self.model.T_cs_des), units.kWh)
        self.model.eta1 = lambda t: self.model.wdot[t]/self.model.x_calc[t] if self.model.x_calc[t] == 0 else 0
        self.model.eta2 = lambda t: self.model.wdot[t]/self.model.x[t] if self.model.x[t] == 0 else 0
        # self.model.pr = lambda t: self.model.Lr*(self.model.xr[t] + self.model.Qin[t]*self.model.frsu[t] + self.model.Qrl*self.model.yrsb[t])
        # self.model.pc = lambda t: self.model.Lc*(self.model.x[t] + self.model.Qc*self.model.ycsu[t])# + Qb*ycsb[t])

        self.model.obj_end_incentive = (
                self.model.D[self.model.t_end] * self.model.disp_time_weighting * self.model.avg_price_disp_storage_incentive *
                (self.model.eta_des) * (
                    self.model.s_calc[self.model.t_end] if self.model.t_end == self.model.t_transition
                        else self.model.s[self.model.t_end]
                )
            )

                
    def addObjective(self):
        def objectiveRule(model):
            return (
                    sum( model.D[t] * 
                    #obj_profit
                    model.Delta[t] * model.P[t] * (model.wdot_s[t] - (model.avg_purchase_price/model.avg_price) * model.wdot_p[t])
                    #obj_cost_cycle_su_hs_sd
                    - (model.Ccsu*model.ycsup[t] + model.Cchsp*model.ychsp[t] + model.alpha*model.ycsdp[t])
                    #obj_cost_cycle_ramping
                    - (model.C_delta_w*(model.wdot_delta_plus[t]+model.wdot_delta_minus[t])+model.C_v_w*(model.wdot_v_plus[t] + model.wdot_v_minus[t]))
                    #obj_cost_rec_su_hs_sd
                    - (model.Crsu*model.yrsup[t] + model.Crhsp*model.yrhsp[t] + model.alpha*(model.yrsb[t]+model.yrsdp[t]))
                    #obj_cost_ops
                    - model.Delta[t]*(model.Cpc*model.wdot[t] + model.Ccsb*model.ycsb[t] + model.Crec*model.xr[t] )
                    for t in model.T)
                    #obj_end_incentive
                    + model.obj_end_incentive
                    )
        def objectiveRuleForceCycle(model):
            return (
                    sum( model.D[t] * 
                    #obj_profit
                    model.Delta[t] * model.P[t] * (model.wdot_s[t] - (model.avg_purchase_price/model.avg_price) * model.wdot_p[t])
                    #obj_cost_cycle_su_hs_sd
                    - (model.Ccsu*model.ycsup[t] + 0.1*model.Cchsp*model.ychsp[t] + model.alpha*model.ycsdp[t])
                    #obj_cost_cycle_ramping
                    - (model.C_delta_w*(model.wdot_delta_plus[t]+model.wdot_delta_minus[t])+model.C_v_w*(model.wdot_v_plus[t] + model.wdot_v_minus[t]))
                    #obj_cost_rec_su_hs_sd
                    - (model.Crsu*model.yrsup[t] + model.Crhsp*model.yrhsp[t] + model.alpha*(model.yrsb[t]+model.yrsdp[t]))
                    #obj_cost_ops
                    - model.Delta[t]*(model.Cpc*model.wdot[t] + model.Ccsb*model.ycsb[t] + model.Crec*model.xr[t] )
                    #obj_force_cycle
                    + model.cycle_incent * (model.ycgb[t] + model.ycge[t])
                    for t in model.T)
                    #obj_end_incentive
                    + model.obj_end_incentive
                    )

        def objectiveRuleSignal(model):
            return (
                    sum(model.D[t] *
                        # obj_profit
                        model.Delta[t] * model.P[t] * (model.wdot_s[t] - (model.avg_purchase_price/model.avg_price) * model.wdot_p[t])
                        # obj_cost_cycle_su_hs_sd
                        - (model.Ccsu * model.ycsup[t] + model.Cchsp * model.ychsp[t] + model.alpha * model.ycsdp[t])
                        # obj_cost_cycle_ramping
                        - (model.C_delta_w * (model.wdot_delta_plus[t] + model.wdot_delta_minus[t]) + model.C_v_w * (
                                model.wdot_v_plus[t] + model.wdot_v_minus[t]))
                        # obj_cost_rec_su_hs_sd
                        - (model.Crsu * model.yrsup[t] + model.Crhsp * model.yrhsp[t] + model.alpha * (
                                model.yrsb[t] + model.yrsdp[t]))
                        # obj_cost_ops
                        - model.Delta[t] * (
                                    model.Cpc * model.wdot[t] + model.Ccsb * model.ycsb[t] + model.Crec * model.xr[t])
                        for t in model.T)
                    - sum(model.Cg_plus[h] * model.g_plus[h] + model.Cg_minus[h] * model.g_minus[h] for h in model.H)
                    # obj_end_incentive
                    + model.obj_end_incentive
            )

        def addObjectiveSimpleReceiver(model):
            return (
                    sum(model.D[t] *
                        # obj_profit
                        model.Delta[t] * model.P[t] * (model.wdot_s[t] - (model.avg_purchase_price / model.avg_price) * model.wdot_p[t])
                        # obj_cost_cycle_su_hs_sd
                        - (model.Ccsu * model.ycsup[t] + model.Cchsp * model.ychsp[t] + model.alpha * model.ycsdp[t])
                        # obj_cost_cycle_ramping
                        - (model.C_delta_w * (model.wdot_delta_plus[t] + model.wdot_delta_minus[t]) + model.C_v_w * (
                                model.wdot_v_plus[t] + model.wdot_v_minus[t]))
                        # obj_cost_ops
                        - model.Delta[t] * (
                                    model.Cpc * model.wdot[t] + model.Ccsb * model.ycsb[t] + model.Crec * model.xr[t])
                        for t in model.T)
                    # obj_end_incentive
                    + model.obj_end_incentive
            )

        def addObjectiveSimpleReceiverSignal(model):
            return (
                    sum(model.D[t] *
                        # obj_profit
                        model.Delta[t] * model.P[t] * (model.wdot_s[t] - (model.avg_purchase_price/model.avg_price) * model.wdot_p[t])
                        # obj_cost_cycle_su_hs_sd
                        - (model.Ccsu * model.ycsup[t] + model.Cchsp * model.ychsp[t] + model.alpha * model.ycsdp[t])
                        # obj_cost_cycle_ramping
                        - (model.C_delta_w * (model.wdot_delta_plus[t] + model.wdot_delta_minus[t]) + model.C_v_w * (
                                model.wdot_v_plus[t] + model.wdot_v_minus[t]))
                        # obj_cost_ops
                        - model.Delta[t] * (
                                    model.Cpc * model.wdot[t] + model.Ccsb * model.ycsb[t] + model.Crec * model.xr[t])
                        for t in model.T)
                    - sum(model.Cg_plus[h] * model.g_plus[h] + model.Cg_minus[h] * model.g_minus[h] for h in model.H)
                    #missed signal penalties
                    - sum(model.Cg_plus[h] * model.g_plus[h] + model.Cg_minus[h] * model.g_minus[h] for h in model.H)
                    # obj_end_incentive
                    + model.obj_end_incentive
            )

        if self.include["force_cycle"]:
            self.model.OBJ = pe.Objective(rule=objectiveRuleForceCycle, sense=pe.maximize)
        elif self.include["signal"]:
            if self.include["simple_receiver"]:
                self.model.OBJ = pe.Objective(rule=addObjectiveSimpleReceiverSignal, sense=pe.maximize)
            else:
                self.model.OBJ = pe.Objective(rule=objectiveRuleSignal, sense=pe.maximize)
        else:
            if self.include["simple_receiver"]:
                self.model.OBJ = pe.Objective(rule=addObjectiveSimpleReceiver, sense=pe.maximize)
            else:
                self.model.OBJ = pe.Objective(rule=objectiveRule, sense=pe.maximize)
            
    def addPersistenceConstraints(self):
        def wdot_s_persist_pos_rule(model, t):
            return model.wdot_s_prev_delta_plus[t] >= model.wdot_s[t] - model.wdot_s_prev[t]

        def wdot_s_persist_neg_rule(model, t):
            return model.wdot_s_prev_delta_minus[t] >= model.wdot_s_prev[t] - model.wdot_s[t]

        self.model.persist_pos_con = pe.Constraint(self.model.T, rule=wdot_s_persist_pos_rule)
        self.model.persist_neg_con = pe.Constraint(self.model.T, rule=wdot_s_persist_neg_rule)

    def addPumpConstraints(self):
        def receiver_pump_rule(model, t, i):
            return model.lr[t] == model.Pr * (model.mdot_r_cs[t] + model.mdot_r_hs[t])

        def convex_cycle_pump1_rule(model, t, i):
            return model.lc[t] >= model.Pc[i] * model.mdot_c[t] + model.Bc[i] * model.y[t]

        def convex_cycle_pump2_rule(model, t, i):
            return model.lc[t] >= model.Pc[i] * model.mdot_c[t] + self.model.Bc[i] * model.ycsu[t]

        def convex_feedwater_pump1_rule(model, t, i):
            return model.lfw[t] >= model.Pfw[i] * model.mdot_c[t] + model.Bfw[i] * model.y[t]

        def convex_feedwater_pump2_rule(model, t, i):
            return model.lfw[t] >= model.Pfw[i] * model.mdot_c[t] + model.Bfw[i] * model.ycsu[t]

        self.model.receiver_pump_con = pe.Constraint(self.model.T_nl * self.model.htf_segments, rule=receiver_pump_rule)
        self.model.convex_cycle_pump1_con = pe.Constraint(self.model.T_nl * self.model.htf_segments,
                                                          rule=convex_cycle_pump1_rule)
        self.model.convex_cycle_pump2_con = pe.Constraint(self.model.T_nl * self.model.htf_segments,
                                                          rule=convex_cycle_pump2_rule)
        self.model.convex_feedwater_pump1_con = pe.Constraint(self.model.T_nl * self.model.fw_segments,
                                                              rule=convex_feedwater_pump1_rule)
        self.model.convex_feedwater_pump2_con = pe.Constraint(self.model.T_nl * self.model.fw_segments,
                                                              rule=convex_feedwater_pump2_rule)


    def addReceiverStartupConstraintsLinear(self):
        def rec_inventory_rule(model, t):
            if t == model.t_start:
                return model.ursu[t] <= model.ursu0 + model.Delta[t]*model.xrsu[t]
            return model.ursu[t] <= model.ursu[t-1] + model.Delta[t]*model.xrsu[t]

        def rec_inv_nonzero_rule(model, t):
            return model.ursu[t] <= model.Er * model.yrsu[t]

        def rec_startup_rule(model, t):
            if t == model.t_start:
                return model.yr[t] <= model.ursu[t]/model.Er + model.yr0 + model.yrsb0
            return model.yr[t] <= model.ursu[t]/model.Er + model.yr[t-1] + model.yrsb[t-1]

        def rec_su_persist_rule(model, t):
            if t == model.t_start:
                return model.yrsu[t] + model.yr0 <= 1
            return model.yrsu[t] +  model.yr[t-1] <= 1

        def ramp_limit_rule(model, t):
            return model.xrsu[t] <= model.Qru*model.yrsu[t]

        def nontrivial_solar_rule(model, t):
            return model.yrsu[t] <= model.Qin[t]

        self.model.rec_inventory_con = pe.Constraint(self.model.T, rule=rec_inventory_rule)
        self.model.rec_inv_nonzero_con = pe.Constraint(self.model.T, rule=rec_inv_nonzero_rule)
        self.model.rec_startup_con = pe.Constraint(self.model.T, rule=rec_startup_rule)
        self.model.rec_su_persist_con = pe.Constraint(self.model.T, rule=rec_su_persist_rule)
        self.model.ramp_limit_con = pe.Constraint(self.model.T, rule=ramp_limit_rule)
        self.model.nontrivial_solar_con = pe.Constraint(self.model.T, rule=nontrivial_solar_rule)

    def addReceiverStartupConstraints(self):
        ### time inventory
        def rec_su_time_inv_rule(model, t):
            if t == self.model.t_start:
                return model.drsu[t] <= model.drsu0 + model.Delta[t]*model.frsu[t]
            return model.drsu[t] <= model.drsu[t-1] + model.Delta[t]*model.frsu[t]

        def rec_su_time_nonzero_rule(model, t):
            return model.drsu[t] <= model.Drsu * model.yrsu[t]

        def rec_startup_time_rule(model, t):
            if t == model.t_start:
                return model.Drsu*model.yr[t] <= model.drsu[t] + model.Drsu*(model.yr0 + model.yrsb0)
            return model.Drsu*model.yr[t] <= model.drsu[t] + model.Drsu*(model.yr[t-1] + model.yrsb[t-1])

        ### energy inventory
        def rec_su_eng_inv1_rule(model, t):
            if t == model.t_start:
                return model.ursu[t] <= model.ursu0 + model.Delta[t]*model.Qin[t]*model.frsu[t]
            return model.ursu[t] <= model.ursu[t-1] + model.Delta[t]*model.Qin[t]*model.frsu[t]

        def rec_su_eng_inv2_rule(model, t):
            if t == model.t_start:
                return model.ursu[t] <= model.ursu0 + model.Delta[t]*model.Qru
            return model.ursu[t] <= model.ursu[t-1] + model.Delta[t]*model.Qru

        def rec_su_eng_nonzero_rule(model, t):
            return model.ursu[t] <= model.Er * model.yrsu[t]

        def rec_startup_eng_rule(model, t):
            if t == model.t_start:
                return model.Er*model.yr[t] <= model.ursu[t] + (model.Er*(model.yr0 + model.yrsb0))
            return model.Er*model.yr[t] <= model.ursu[t] + (model.Er*(model.yr[t-1] + model.yrsb[t-1]))

        #binary logic
        def rec_startup_frac_nonzero_rule(model, t):
            return model.frsu[t] <= model.yrsu[t]

        def rec_force_startup_frac_rule(model, t):
            return model.frsu[t] >= model.yrsu[t] - model.yr[t]

        self.model.rec_su_time_inv_con = pe.Constraint(self.model.T, rule=rec_su_time_inv_rule)
        self.model.rec_su_time_nonzero_con = pe.Constraint(self.model.T, rule=rec_su_time_nonzero_rule)
        self.model.rec_startup_time_con = pe.Constraint(self.model.T, rule=rec_startup_time_rule)
        self.model.rec_su_eng_inv1_con = pe.Constraint(self.model.T, rule=rec_su_eng_inv1_rule)
        self.model.rec_su_eng_inv2_con = pe.Constraint(self.model.T, rule=rec_su_eng_inv2_rule)
        self.model.rec_su_eng_nonzero_con = pe.Constraint(self.model.T, rule=rec_su_eng_nonzero_rule)
        self.model.rec_startup_eng_con = pe.Constraint(self.model.T, rule=rec_startup_eng_rule)
        self.model.rec_startup_frac_nonzero_con = pe.Constraint(self.model.T, rule=rec_startup_frac_nonzero_rule)
        self.model.rec_force_startup_frac_con = pe.Constraint(self.model.T, rule=rec_force_startup_frac_rule)

    #Receiver Collection
    def addReceiverSupplyAndDemandConstraints(self):
        def rec_production_rule(model, t):
            #return model.xr[t] + model.xrsu[t] + model.Qrsd*model.yrsd[t] <= model.Qin[t]
            return model.xr[t] <= model.Qin[t] * (1 - model.frsu[t] - model.frsd[t])

        def rec_generation_rule(model, t):
            return model.xr[t] <= model.Qin[t] * model.yr[t]

        def min_generation_rule(model, t):
            return model.xr[t] >= model.Qrl * (model.yr[t] - model.frsu[t] - model.frsd[t])

        self.model.rec_production_con = pe.Constraint(self.model.T, rule=rec_production_rule)
        self.model.rec_generation_con = pe.Constraint(self.model.T, rule=rec_generation_rule)
        self.model.min_generation_con = pe.Constraint(self.model.T, rule=min_generation_rule)

    # Receiver Shutdown
    def addReceiverShutdownConstraints(self):
        ### time inventory
        def rec_sd_time_inv_rule(model, t):
            if t == model.t_start:
                return model.drsd[t] <= model.drsd0 + model.Delta[t]*model.frsd[t]
            return model.drsd[t] <= model.drsd[t-1] + model.Delta[t]*model.frsd[t]

        def rec_sd_time_nonzero_rule(model, t):
            return model.drsd[t] <= model.Drsd * model.yrsd[t]

        def rec_shutdown_time_rule(model, t):
            if t == model.t_start:
                return model.Drsd*model.yrsd[t] >= model.yrsd0*model.Drsd - model.drsd0
            return model.Drsd*model.yrsd[t] >= model.yrsd[t-1]*model.Drsd - model.drsd[t-1]

        ### energy inventory
        def rec_sd_eng_inv_rule(model, t):
            if t == model.t_start:
                return model.ursd[t] <= model.ursd0 + model.Delta[t]*model.Qin[t]*model.frsd[t]
            return model.ursd[t] <= model.ursd[t-1] + model.Delta[t]*model.Qin[t]*model.frsd[t]

        def rec_sd_eng_nonzero_rule(model, t):
            return model.ursd[t] <= model.Qrsd * model.yrsd[t];

        def rec_shutdown_eng_rule(model, t):
            if t == model.t_start:
                return model.Qrsd*model.yrsd[t] >= model.yrsd0*model.Qrsd - model.ursd0
            return model.Qrsd*model.yrsd[t] >= model.yrsd[t-1]*model.Qrsd - model.ursd[t-1]

        #binary logic
        def rec_sd_frac_nonzero_rule(model, t):
            return model.frsd[t] <= model.yrsd[t]

        def rec_sd_frac_force_rule(model, t):
            return model.frsd[t] >= model.yrsd[t] - model.yr[t]

        def rec_shutdown_rule(model, t):
            if t == model.t_end:
                return pe.Constraint.Feasible
            return model.yrsd[t] >= (model.yr[t] - model.yr[t+1]) + (model.yrsb[t] - model.yrsb[t+1]);

        self.model.rec_sd_time_inv_con = pe.Constraint(self.model.T, rule=rec_sd_time_inv_rule)
        self.model.rec_sd_time_nonzero_con = pe.Constraint(self.model.T, rule=rec_sd_time_nonzero_rule)
        self.model.rec_shutdown_time_con = pe.Constraint(self.model.T, rule=rec_shutdown_time_rule)
        self.model.rec_sd_eng_inv_con = pe.Constraint(self.model.T, rule=rec_sd_eng_inv_rule)
        self.model.rec_sd_eng_nonzero_con = pe.Constraint(self.model.T, rule=rec_sd_eng_nonzero_rule)
        self.model.rec_shutdown_eng_con = pe.Constraint(self.model.T, rule=rec_shutdown_eng_rule)
        self.model.rec_sd_frac_nonzero_con = pe.Constraint(self.model.T, rule=rec_sd_frac_nonzero_rule)
        self.model.rec_sd_frac_force_con = pe.Constraint(self.model.T, rule=rec_sd_frac_force_rule)
        self.model.rec_shutdown_con = pe.Constraint(self.model.T, rule=rec_shutdown_rule)

    def addReceiverPenaltyConstraints(self):
        def rec_su_pen_rule(model, t):
            if t == model.t_start:
                return self.model.yrsup[t] >= self.model.yrsu[t] - self.model.yrsu0
            return self.model.yrsup[t] >= self.model.yrsu[t] - self.model.yrsu[t-1]

        def rec_hs_pen_rule(model, t):
            if t == model.t_start:
                return self.model.yrhsp[t] >= self.model.yr[t] - (1 - self.model.yrsb0)
            return self.model.yrhsp[t] >= self.model.yr[t] - (1 - self.model.yrsb[t-1])

        def rec_sd_pen_rule(model, t):
            if t == model.t_start:
                return self.model.yrsdp[t] >= self.model.yrsd0 - self.model.yrsd[t]
            return self.model.yrsdp[t] >= self.model.yrsd[t-1] - self.model.yrsd[t]

        self.model.rec_su_pen_con = pe.Constraint(self.model.T, rule=rec_su_pen_rule)
        self.model.rec_hs_pen_con = pe.Constraint(self.model.T, rule=rec_hs_pen_rule)
        self.model.rec_sd_pen_con = pe.Constraint(self.model.T, rule=rec_sd_pen_rule)

    def addReceiverModeLogicConstraints(self):
        def su_sd_nontrivial_solar_rule(model, t):
            if model.Qin[t].value > 0:
                return model.yrsu[t] + model.yrsd[t] <= 1
            return model.yrsu[t] + model.yrsd[t] <= 0

        def rec_su_run_persist_rule(model, t):
            if t == model.t_start:
                return model.yrsu[t] + model.yr0 <= 1
            return model.yrsu[t] + model.yr[t-1] <= 1

        def rec_su_sb_persist_rule(model, t):
            if t == model.t_start:
                return model.yrsu[t] + model.yrsb0 <= 1
            return model.yrsu[t] + model.yrsb[t-1] <= 1

        def rec_su_sb_sd_pack_rule(model, t):
            return model.yrsu[t] + model.yrsb[t] + model.yrsd[t] <= 1

        def rec_gen_persist_rule(model, t):
            return model.Qrl * model.yr[t] <= model.Qin[t]

        def rec_sb_run_pack_rule(model, t):
            return model.yr[t] + model.yrsb[t] <= 1

        def rsb_persist_rule(model, t):
            if t == model.t_start:
                return model.yrsb[t] <= model.yr0 + model.yrsb0
            return model.yrsb[t] <= model.yr[t-1] + model.yrsb[t-1]

        def rec_force_off_rule(model, t):
            if t == model.t_start:
                return model.yr[t] + model.yrsd0 <= 1
            return model.yr[t] + model.yrsd[t-1] <= 1

        self.model.su_sd_nontrivial_solar_con = pe.Constraint(self.model.T, rule=su_sd_nontrivial_solar_rule)
        self.model.rec_su_run_persist_con = pe.Constraint(self.model.T, rule=rec_su_run_persist_rule)
        self.model.rec_su_sb_persist_con = pe.Constraint(self.model.T, rule=rec_su_sb_persist_rule)
        self.model.rec_su_sb_sd_pack_con = pe.Constraint(self.model.T, rule=rec_su_sb_sd_pack_rule)
        self.model.rec_gen_persist_con = pe.Constraint(self.model.T, rule=rec_gen_persist_rule)
        self.model.rec_sb_run_pack_con = pe.Constraint(self.model.T, rule=rec_sb_run_pack_rule)
        self.model.rsb_persist_con = pe.Constraint(self.model.T, rule=rsb_persist_rule)
        self.model.rec_force_off_con = pe.Constraint(self.model.T, rule=rec_force_off_rule)

    def addReceiverMassFlowRateConstraints(self):
        def mdot_r_upper1_rule(model, t):
            return model.mdot_r_cs[t] + model.mdot_r_hs[t] <= model.mdot_r_max*(model.yrsu[t] + model.yr[t] + model.yrsb[t])

        def mdot_r_upper2_rule(model, t):
            return model.mdot_r_cs[t] + model.mdot_r_hs[t] <= model.mdot_r_max

        def mdot_r_lower1_rule(model, t):
            return model.mdot_r_cs[t] + model.mdot_r_hs[t] >= model.mdot_r_min*(model.yr[t] + model.yrsb[t] - model.frsd[t])

        def mdot_r_lower2_rule(model, t):
            return model.mdot_r_cs[t] + model.mdot_r_hs[t] >= model.mdot_r_min*(model.frsu[t])

        def mdot_r_upper3_rule(model, t):
            return model.mdot_r_cs[t] <= model.mdot_r_max * (model.yrsu[t] + model.yrsb[t])

        def mdot_r_upper4_rule(model, t):
            return model.mdot_r_hs[t] <= model.mdot_r_max * model.yr[t]

        self.model.mdot_r_upper1_con = pe.Constraint(self.model.T_nl, rule=mdot_r_upper1_rule)
        self.model.mdot_r_upper2_con = pe.Constraint(self.model.T_nl, rule=mdot_r_upper2_rule)
        self.model.mdot_r_lower1_con = pe.Constraint(self.model.T_nl, rule=mdot_r_lower1_rule)
        self.model.mdot_r_lower2_con = pe.Constraint(self.model.T_nl, rule=mdot_r_lower2_rule)
        self.model.mdot_r_upper3_con = pe.Constraint(self.model.T_nl, rule=mdot_r_upper3_rule)
        self.model.mdot_r_upper4_con = pe.Constraint(self.model.T_nl, rule=mdot_r_upper4_rule)

    def addReceiverTemperatureConstraints(self):
        def T_rout_lower1_rule(model, t):
            return model.T_rout[t] >= model.T_cs_min*(model.yr[t] + model.yrsb[t])

        def T_rout_lower2_rule(model, t):
            return model.T_rout[t] >= model.T_cs_min*model.yrsu[t]

        def T_rout_upper1_rule(model, t):
            return model.T_rout[t] <= model.T_rout_max*(model.yrsu[t] + model.yr[t] + model.yrsb[t])

        def T_rout_upper2_rule(model, t):
            return model.T_rout[t] <= model.T_rout_max

        self.model.T_rout_lower1_con = pe.Constraint(self.model.T_nl, rule=T_rout_lower1_rule)
        self.model.T_rout_lower2_con = pe.Constraint(self.model.T_nl, rule=T_rout_lower2_rule)
        self.model.T_rout_upper1_con = pe.Constraint(self.model.T_nl, rule=T_rout_upper1_rule)
        self.model.T_rout_upper2_con = pe.Constraint(self.model.T_nl, rule=T_rout_upper2_rule)

    def addReceiverPowerBalanceConstraints(self):
        def rec_power_bal_rule(model, t):
            return model.xr[t] - model.Qrsb*model.yrsb[t] == model.Cp*(model.mdot_r_hs[t]*model.T_rout[t] +
                    model.mdot_r_cs[t]*model.T_rout[t] - model.mdot_r_hs[t]*model.T_cs[t] -
                    model.mdot_r_cs[t]*model.T_cs[t])

        def rec_clr_sky_control_rule(model, t):
            return model.xr[t] <= model.F[t]*model.Cp*(model.mdot_r_hs[t]*model.T_rout_max +
                    model.mdot_r_cs[t]*model.T_rout_max  - model.mdot_r_hs[t]*model.T_cs[t] -
                    model.mdot_r_cs[t]*model.T_cs[t])
        self.model.rec_power_bal_con = pe.Constraint(self.model.T_nl, rule=rec_power_bal_rule)
        self.model.rec_clr_sky_control_con = pe.Constraint(self.model.T_nl, rule=rec_clr_sky_control_rule)

    def addSimpleReceiverConstraint(self):
        def simple_receiver_rule(model, t):
            return model.xr[t] <= model.Qin[t]

        self.model.simple_receiver_con = pe.Constraint(self.model.T, rule=simple_receiver_rule)

    def addTESEnergyBalanceConstraints(self):
        def tes_balance_rule(model, t):
            if t == model.t_transition+1:
                return model.s[t] - model.s0 == model.Delta[t] * (model.xr[t] - (model.Qc*model.ycsu[t] + model.x[t] + model.Qrsb*model.yrsb[t]))
            return model.s[t] - model.s[t-1] == model.Delta[t] * (model.xr[t] - (model.Qc*model.ycsu[t] + model.x[t] + model.Qrsb*model.yrsb[t]))

        def tes_balance_simple_receiver_rule(model, t):
            if t == model.t_transition+1:
                return model.s[t] - model.s0 == model.Delta[t] * (model.xr[t] - (model.Qc*model.ycsu[t] + model.x[t]))
            return model.s[t] - model.s[t-1] == model.Delta[t] * (model.xr[t] - (model.Qc*model.ycsu[t] + model.x[t]))

        # def tes_start_up_rule(model, t):
        #    if t == model.t_start:
        #        return model.s0 >= model.Delta[t]*model.delta_rs[t]*( (model.Qu + model.Qb)*( -3 + model.yrsu[t] + model.y0 + model.y[t] + model.ycsb0 + model.ycsb[t] ) + model.x[t] + model.Qb*model.ycsb[t] )
        #    return model.s[t-1] >= model.Delta[t]*model.delta_rs[t]*( (model.Qu + model.Qb)*( -3 + model.yrsu[t] + model.y[t-1] + model.y[t] + model.ycsb[t-1] + model.ycsb[t] ) + model.x[t] + model.Qb*model.ycsb[t] )

        # def maintain_tes_rule(model):
        #     return model.s[model.num_periods] >= model.s0
        if self.include["simple_receiver"]:
            self.model.tes_balance_con = pe.Constraint(self.model.T_l, rule=tes_balance_simple_receiver_rule)
        else:
            self.model.tes_balance_con = pe.Constraint(self.model.T_l, rule=tes_balance_rule)
        #self.model.tes_start_up_con = pe.Constraint(self.model.T, rule=tes_start_up_rule)
        #self.model.maintain_tes_con = pe.Constraint(rule=maintain_tes_rule)  Used?

    def addThermalStorageMassTempConstraints(self):
        ### mass balance
        def cold_side_mass_rule(model, t):
            if t == model.t_start:
                return model.mass_cs[t] - model.mass_cs0 == units.convert(model.Delta[t], units.s)*(model.mdot_c[t] - model.mdot_r_hs[t])
            return model.mass_cs[t] - model.mass_cs[t-1] == units.convert(model.Delta[t], units.s)*(model.mdot_c[t] - model.mdot_r_hs[t])

        def hot_side_mass_rule(model, t):
            if t == model.t_start:
                return model.mass_hs[t] - model.mass_hs0 == units.convert(model.Delta[t], units.s)*(model.mdot_r_hs[t] - model.mdot_c[t])
            return model.mass_hs[t] - model.mass_hs[t-1] == units.convert(model.Delta[t], units.s)*(model.mdot_r_hs[t] - model.mdot_c[t])

        ### energy balance
        def cold_side_energy_balance_rule(model, t):
            if t == model.t_start:
                return (
                model.mass_cs[t] * model.T_cs[t] - (model.mass_cs0 * model.T_cs0) ==
                units.convert(model.Delta[t], units.s) * (model.mdot_r_cs[t] * model.T_rout[t] + model.mdot_c[t] * model.T_cout[t] - (
                            model.mdot_r_cs[t] * model.T_cs[t] + model.mdot_r_hs[t] * model.T_cs[t]))
                )
            return (
                model.mass_cs[t] * model.T_cs[t] - (model.mass_cs[t-1] * model.T_cs[t-1]) ==
                units.convert(model.Delta[t], units.s) * (model.mdot_r_cs[t] * model.T_rout[t] + model.mdot_c[t] * model.T_cout[t] - (
                        model.mdot_r_cs[t] * model.T_cs[t] + model.mdot_r_hs[t] * model.T_cs[t]))
            )

        def hot_side_energy_balance_rule(model, t):
            if t == model.t_start:
                return (
                    model.mass_hs[t] * model.T_hs[t] - (model.mass_hs0 * model.T_hs0) ==
                    units.convert(model.Delta[t], units.s) * (model.mdot_r_hs[t] * model.T_rout[t] - model.mdot_c[t] * model.T_hs[t])
                )
            return (
                model.mass_hs[t] * model.T_hs[t] - (model.mass_hs[t-1] * model.T_hs[t-1]) ==
                units.convert(model.Delta[t], units.s) * (model.mdot_r_hs[t] * model.T_rout[t] - model.mdot_c[t] * model.T_hs[t])
            )

        self.model.cold_side_mass_con = pe.Constraint(self.model.T_nl, rule=cold_side_mass_rule)
        self.model.hot_side_mass_con = pe.Constraint(self.model.T_nl, rule=hot_side_mass_rule)
        self.model.cold_side_energy_balance_con = pe.Constraint(self.model.T_nl, rule=cold_side_energy_balance_rule)
        self.model.hot_side_energy_balance_con = pe.Constraint(self.model.T_nl, rule=hot_side_energy_balance_rule)


    def addCycleStartupConstraints(self):
        # def pc_inventory_rule(model, t):  replaced by pc_su_eng_inv_rule
        #     if t == model.t_start:
        #         return model.ucsu[t] <= model.ucsu0 + model.Delta[t] * model.Qc[t] * model.ycsu[t]
        #     return model.ucsu[t] <= model.ucsu[t-1] + model.Delta[t] * model.Qc[t] * model.ycsu[t]

        def pc_su_eng_inv_rule(model, t):
            if t == model.t_start:
                return model.ucsu[t] <= model.ucsu0 + model.Delta[t] * model.Qc * model.ycsu[t] + (model.Ec - model.Ew) * model.ycsb0
            return model.ucsu[t] <= model.ucsu[t-1] + model.Delta[t] * model.Qc * model.ycsu[t] + (model.Ec - model.Ew) * model.ycsb[t-1]

        def pc_su_eng_nonzero_rule(model, t):
            return model.ucsu[t] <= model.Ec * model.ycsu[t]

        def pc_startup_rule(model, t):
            if t == model.t_start:
                return model.Ec*model.y[t] <= model.ucsu0 + (model.Ec*model.y0)
            return model.Ec*model.y[t] <= model.ucsu[t-1] + (model.Ec*model.y[t-1])
            # if model.Delta[t] >= 1 and t == model.t_start:
            #     return model.y[t] <= model.ucsu[t]/model.Ec + model.y0 + model.ycsb0
            # elif model.Delta[t] >= 1 and t > 1:
            #     return model.y[t] <= model.ucsu[t]/model.Ec + model.y[t-1] + model.ycsb[t-1]
            # elif model.Delta[t] < 1 and t == model.t_start:
            #     return model.y[t] <= model.ucsu0/model.Ec + model.y0 + model.ycsb0
            # # only case remaining: Delta[t]<1, t>1
            # return model.y[t] <= model.ucsu[t-1]/model.Ec + model.y[t-1] + model.ycsb[t-1]

        def cycle_temp_su_upper_rule(model, t):
            return model.Qc*model.ycsu[t] <= model.Cp*(model.mdot_c[t]*model.T_hs[t] - model.mdot_c[t]*model.T_cout[t]) + model.Qu * (1 - model.ycsu[t])

        def cycle_temp_su_lower_rule(model, t):
            return model.Qc*model.ycsu[t] >= model.Cp*(model.mdot_c[t]*model.T_hs[t] - model.mdot_c[t]*model.T_cout[t]) - model.Qu * (1 - model.ycsu[t])

        """replaced by penalty rules within addPowerCycleThermalInputConstraints"""
        # def pc_production_rule(model, t):
        #     return model.x[t] + model.Qc[t]*model.ycsu[t] <= model.Qu
        # def pc_min_gen_rule(model, t):
        #     return model.x[t] >= model.Ql * model.y[t]
        
        # self.model.pc_inventory_con = pe.Constraint(self.model.T, rule=pc_inventory_rule)
        self.model.pc_su_eng_inv_con = pe.Constraint(self.model.T, rule=pc_su_eng_inv_rule)
        self.model.pc_su_eng_nonzero_con = pe.Constraint(self.model.T, rule=pc_su_eng_nonzero_rule)
        self.model.pc_startup_con = pe.Constraint(self.model.T, rule=pc_startup_rule)
        self.model.cycle_temp_su_upper_con = pe.Constraint(self.model.T_nl, rule=cycle_temp_su_upper_rule)
        self.model.cycle_temp_su_lower_con = pe.Constraint(self.model.T_nl, rule=cycle_temp_su_lower_rule)


    def addPowerCycleThermalInputConstraints(self):
        def pc_input_nonzero_rule(model, t):
            return model.x[t] <= model.Qu * model.y[t]

        def pc_upper_input_pen_rule(model, t):
            if model.t_transition == 0:
                return model.x[t] <= model.Qu - model.ku * (model.T_cin_design - model.T_hs0)
            return model.x[t] <= model.Qu - model.ku * (model.T_cin_design - model.T_hs[model.t_transition])

        def pc_lower_input_pen_nonzero_rule(model, t):
            if model.t_transition == 0:
                return model.x[t] >= model.Ql * model.y[t] - model.kl * (model.T_cin_design - model.T_hs0)
            return model.x[t] >= model.Ql * model.y[t] - model.kl * (
                    model.T_cin_design - model.T_hs[model.t_transition])

        def cycle_temp_prod_lower_rule(model, t):
            return (
                    model.T_hs[t] - model.T_cout[t] >= (model.alpha_b + model.alpha_T * model.T_hs[t] +
                        model.alpha_m * model.mdot_c[t])
                    - model.delta_T_max * (1 - model.y[t])
            )

        def cycle_temp_prod_upper_rule(model, t):
            return (
                    model.T_hs[t] - model.T_cout[t] <= (model.alpha_b + model.alpha_T * model.T_hs[t] +
                        model.alpha_m * model.mdot_c[t])
                    + model.delta_T_max * (1 - model.y[t])
            )

        self.model.pc_input_nonzero_con = pe.Constraint(self.model.T_l, rule=pc_input_nonzero_rule)
        self.model.pc_upper_input_pen_con = pe.Constraint(self.model.T_l, rule=pc_upper_input_pen_rule)
        self.model.pc_lower_input_pen_nonzero_con = pe.Constraint(self.model.T_l, rule=pc_lower_input_pen_nonzero_rule)
        self.model.cycle_temp_prod_lower_con = pe.Constraint(self.model.T_nl, rule=cycle_temp_prod_lower_rule)
        self.model.cycle_temp_prod_upper_con = pe.Constraint(self.model.T_nl, rule=cycle_temp_prod_upper_rule)

    def addPowerCycleMassFlowRateConstraints(self):
        def mdot_c_upper1_rule(model, t):
            return model.mdot_c[t] <= model.mdot_c_max*(model.y[t] + model.ycsu[t])

        def mdot_c_upper2_rule(model, t):
            return model.mdot_c[t] <= model.mdot_c_max

        def mdot_c_lower1_rule(model, t):
            return model.mdot_c[t] >= model.mdot_c_min*(model.y[t])

        self.model.mdot_c_upper1_con = pe.Constraint(self.model.T_nl, rule=mdot_c_upper1_rule)
        self.model.mdot_c_upper2_con = pe.Constraint(self.model.T_nl, rule=mdot_c_upper2_rule)
        self.model.mdot_c_lower1_con = pe.Constraint(self.model.T_nl, rule=mdot_c_lower1_rule)

    def addPowerCycleTemperatureConstraints(self):
        def T_cout_upper1_rule(model, t):
            return model.T_cout[t] <= model.T_cs_max * (model.ycsu[t] + model.y[t])

        def T_cout_lower1_rule(model, t):
            return model.T_cout[t] >= model.T_cs_min * model.y[t]

        def T_cout_lower2_rule(model, t):
            return model.T_cout[t] >= model.T_cs_min * model.ycsu[t]

        self.model.T_cout_upper1_con = pe.Constraint(self.model.T_nl, rule=T_cout_upper1_rule)
        self.model.T_cout_lower1_con = pe.Constraint(self.model.T_nl, rule=T_cout_lower1_rule)
        self.model.T_cout_lower2_con = pe.Constraint(self.model.T_nl, rule=T_cout_lower2_rule)

    def addPowerCycleEnergyOutputConstraints(self):
        ### linear power relation
        def cycle_output_linear_rule(model, t):
            return model.wdot[t] == (model.etaamb[t]/model.eta_des) * (model.etap*model.x[t] + model.y[t]*(model.Wdotu - model.etap*model.Qu))

        ### variable bounds, nonzero, and wdot ramping
        def power_ub_rule(model, t):
            return model.wdot[t] <= model.Wdotu*(model.etaamb[t]/model.eta_des)*model.y[t]

        def power_lb_rule(model, t):
            return model.wdot[t] >= model.Wdotl*(model.etaamb[t]/model.eta_des)*model.y[t]

        ### non-linear power regression
        def cycle_power_nonlinear_upper_rule(model, t):
            return model.wdot[t] <= (model.etaamb[t]/model.eta_des) * (
                    (model.beta_b + model.beta_T*model.T_hs[t]  +
                    model.beta_m*model.mdot_c[t] + model.beta_mT*model.mdot_c[t]*model.T_hs[t]
                    ) +
                    model.Wdot_design * (1 - model.y[t]))

        def cycle_power_nonlinear_lower_rule(model, t):
            return model.wdot[t] >= (model.etaamb[t]/model.eta_des) * (
                    (model.beta_b + model.beta_T*model.T_hs[t]  +
                    model.beta_m*model.mdot_c[t] + model.beta_mT*model.mdot_c[t]*model.T_hs[t]
                     ) -
                    model.Wdot_design * (1 - model.y[t])
                )

        self.model.cycle_output_linear_con = pe.Constraint(self.model.T_l, rule=cycle_output_linear_rule)
        self.model.power_ub_con = pe.Constraint(self.model.T, rule=power_ub_rule)
        self.model.power_lb_con = pe.Constraint(self.model.T, rule=power_lb_rule)
        self.model.cycle_power_nonlinear_upper_con = pe.Constraint(self.model.T_nl, rule=cycle_power_nonlinear_upper_rule)
        self.model.cycle_power_nonlinear_lower_con = pe.Constraint(self.model.T_nl, rule=cycle_power_nonlinear_lower_rule)

    def addPowerCycleOutputRampingConstraints(self):
        def change_in_w_pos_rule(model, t):
            if t == model.t_start:
                return model.wdot_delta_plus[t] >= model.wdot[t] - model.wdot0
            return model.wdot_delta_plus[t] >= model.wdot[t] - model.wdot[t-1]

        def change_in_w_neg_rule(model, t):
            if t == model.t_start:
                return model.wdot_delta_minus[t] >= model.wdot0 - model.wdot[t]
            return model.wdot_delta_minus[t] >= model.wdot[t-1] - model.wdot[t]

        def cycle_ramp_rate_pos_rule(model, t):
            if t > model.t_start:
                return pe.Constraint.Feasible
            return (
                    model.wdot_delta_plus[t] - model.wdot_v_plus[t] <= model.W_delta_plus*model.Delta[t]
                    + ((model.etaamb[t]/model.eta_des)*model.W_u_plus[t] - model.W_delta_plus*model.Delta[t]) * model.ycgb[t]
            )

        def cycle_ramp_rate_neg_rule(model, t):
            return (
                    model.wdot_delta_minus[t] - model.wdot_v_minus[t] <= model.W_delta_minus*model.Delta[t]
                    + ((model.etaamb[t]/model.eta_des)*model.W_u_minus[t] - model.W_delta_minus*model.Delta[t]) * model.ycge[t]
            )

        self.model.change_in_w_pos_con = pe.Constraint(self.model.T, rule=change_in_w_pos_rule)
        self.model.change_in_w_neg_con = pe.Constraint(self.model.T, rule=change_in_w_neg_rule)
        self.model.cycle_ramp_rate_pos_con = pe.Constraint(self.model.T, rule=cycle_ramp_rate_pos_rule)
        self.model.cycle_ramp_rate_neg_con = pe.Constraint(self.model.T, rule=cycle_ramp_rate_neg_rule)

    def addElectricBalanceConstraints(self):
        def grid_max_rule(model, t):
            return model.wdot_s[t] <= model.Wdotnet[t]

        def grid_sun_rule(model, t):
            if t > model.t_transition:
                return (
                    model.wdot_s[t] - model.wdot_p[t] == (1 - model.etac[t]) * model.wdot[t]
                    - (model.Lr*(model.xr[t] + model.Qin[t]*model.frsu[t] + model.Qrl*model.yrsb[t]))  # model.pr[t]
                    - (model.Ehs / model.Delta[t]) * (model.yrsu[t] + 2 * model.yrhsp[t] + model.yrsdp[t])
                    - (model.Wh_track - model.Wh_comm) * (model.yrsu[t] + model.yr[t] + model.yrsb[t]) - model.Wh_comm
                    - model.Wht_part * model.yr[t] - model.Wht_full * (1 - model.yr[t])
                     - model.Lc*(model.x[t] + model.Qc*model.ycsu[t])  #model.pc[t]
                    - model.Wb * (model.ycsb[t] + model.ycsu[t]) - model.Wc * (1 - model.y[t])
                )
            return (
                model.wdot_s[t] - model.wdot_p[t] == (1 - model.etac[t]) * model.wdot[t] - model.lr[t]
                - (model.Ehs / model.Delta[t]) * (model.yrsu[t] + 2 * model.yrhsp[t] + model.yrsdp[t])
                - (model.Wh_track - model.Wh_comm) * (model.yrsu[t] + model.yr[t] + model.yrsb[t]) - model.Wh_comm
                - model.Wht_part * model.yr[t] - model.Wht_full * (1 - model.yr[t]) - (model.lc[t] + model.lfw[t])
                - model.Wb * (model.ycsb[t] + model.ycsu[t]) - model.Wc * (1 - model.y[t])
            )

        def simple_grid_sun_rule(model, t): #power cycle losses from grid_sun, plus P_field_rec
            return (
                    model.wdot_s[t] - model.wdot_p[t] == (1 - model.etac[t]) * model.wdot[t] - model.P_field_rec[t]
                    - model.Lc * (model.x[t] + model.Qc * model.ycsu[t])  # model.pc[t]
                    - model.Wb * (model.ycsb[t] + model.ycsu[t]) - model.Wc * (1 - model.y[t])
            )

        def sell_production_rule(model, t):
            return model.wdot_s[t] <= model.wdot[t]

        def purchase_nonzero_rule(model, t):
            return model.wdot_p[t] <= model.Wdot_p_max*(1-model.y[t])

        self.model.grid_max_con = pe.Constraint(self.model.T, rule=grid_max_rule)
        if self.include["simple_receiver"]:
            self.model.simple_grid_sun_con = pe.Constraint(self.model.T, rule=simple_grid_sun_rule)
        else:
            self.model.grid_sun_con = pe.Constraint(self.model.T, rule=grid_sun_rule)
        self.model.sell_production_con = pe.Constraint(self.model.T, rule=sell_production_rule)
        self.model.purchase_nonzero_con = pe.Constraint(self.model.T, rule=purchase_nonzero_rule)

    def addMinUpAndDowntimeConstraints(self):
        def min_cycle_uptime_rule(model, t):
            if pe.value(model.Delta_e[t] > (model.Yu - model.Yu0) * model.y0):
                return sum(model.ycgb[tp] for tp in model.T if pe.value(model.Delta_e[t]-model.Delta_e[tp] < model.Yu) and pe.value(model.Delta_e[t] - model.Delta_e[tp] >= 0)) <= model.y[t]
            return pe.Constraint.Feasible

        def min_cycle_downtime_rule(model, t):
            if pe.value(model.Delta_e[t] > ((model.Yd - model.Yd0)*(1-model.y0))):
                return sum( model.ycge[tp] for tp in model.T if pe.value(model.Delta_e[t]-model.Delta_e[tp] < model.Yd) and pe.value(model.Delta_e[t] - model.Delta_e[tp] >= 0))  <= (1 - model.y[t])
            return pe.Constraint.Feasible

        def cycle_start_end_gen_rule(model, t):
            if t == model.t_start:
                return model.ycgb[t] - model.ycge[t] == model.y[t] - model.y0
            return model.ycgb[t] - model.ycge[t] == model.y[t] - model.y[t-1]

        def cycle_min_updown_init_rule(model, t):
            if pe.value(model.Delta_e[t] <= max(pe.value(model.y0*(model.Yu-model.Yu0)), pe.value((1-model.y0)*(model.Yd-model.Yd0)))):
                return model.y[t] == model.y0
            return pe.Constraint.Feasible
        
        self.model.min_cycle_uptime_con = pe.Constraint(self.model.T, rule=min_cycle_uptime_rule)
        self.model.min_cycle_downtime_con = pe.Constraint(self.model.T, rule=min_cycle_downtime_rule)
        self.model.cycle_start_end_gen_con = pe.Constraint(self.model.T, rule=cycle_start_end_gen_rule)
        self.model.cycle_min_updown_init_con = pe.Constraint(self.model.T, rule=cycle_min_updown_init_rule)

    def addPowerCyclePenaltyConstraints(self):
        def cycle_start_pen_rule(model, t):
            if t == model.t_start:
                return model.ycsup[t] >= model.ycsu[t] - model.ycsu0
            return model.ycsup[t] >= model.ycsu[t] - model.ycsu[t - 1]

        def cycle_sb_pen_rule(model, t):
            if t == model.t_start:
                 return model.ychsp[t] >= model.y[t] - (1 - model.ycsb0)
            return model.ychsp[t] >= model.y[t] - (1 - model.ycsb[t-1])

        def cycle_shutdown_rule(model, t):
            if t == model.t_start:
                return model.ycsd[t] >= model.y0 - model.y[t] + model.ycsb0 - model.ycsb[t]
            return model.ycsd[t] >= model.y[t-1] - model.y[t] + model.ycsb[t-1] - model.ycsb[t]

        self.model.cycle_start_pen_con = pe.Constraint(self.model.T, rule=cycle_start_pen_rule)
        self.model.cycle_sb_pen_con = pe.Constraint(self.model.T, rule=cycle_sb_pen_rule)
        self.model.cycle_shutdown_con = pe.Constraint(self.model.T, rule=cycle_shutdown_rule)
        
    def addCycleLogicConstraints(self):
        def pc_su_sb_op_pack_rule(model, t):
            return model.ycsu[t] + model.y[t] + model.ycsb[t] <= 1

        def pc_su_persist_rule(model, t):
            if t == model.t_start:
                return model.ycsu[t] + model.y0 <= 1
            return model.ycsu[t] + model.y[t-1] <= 1

        def pc_sb_start_rule(model, t):
            if t == model.t_start:
                return model.ycsb[t] <= model.y0 + model.ycsb0
            return model.ycsb[t] <= model.y[t-1] + model.ycsb[t-1]

        self.model.pc_su_sb_op_pack_con = pe.Constraint(self.model.T, rule=pc_su_sb_op_pack_rule)
        self.model.pc_su_persist_con = pe.Constraint(self.model.T, rule=pc_su_persist_rule)
        self.model.pc_sb_start_con = pe.Constraint(self.model.T, rule=pc_sb_start_rule)
        
    def addPVConstraints(self):
        def pv_batt_lim_rule(model, t):
            return model.wbc_pv[t] <= model.wpv[t]

        def pv_DC_lim_rule(model, t):
            return model.wpv[t] <= model.wpv_dc[t]*model.ypv[t]

        def inv_clipping_DC_rule(model, t):
            return model.wpv[t] - model.wbc_pv[t] <= model.Winv_lim*model.ypv[t]
        
        self.model.pv_DC_lim_con = pe.Constraint(self.model.T, rule=pv_DC_lim_rule)
        if self.include["battery"]:        
            self.model.pv_batt_lim_con = pe.Constraint(self.model.T, rule=pv_batt_lim_rule)
            self.model.inv_clipping_DC_con = pe.Constraint(self.model.T, rule=inv_clipping_DC_rule)
        
    def addBatteryConstraints(self):
        def battery_balance_rule(model, t):
            if t == model.t_start:
                return model.soc[t] == model.soc0 + model.Delta[t]*(model.i_p[t] - model.i_n[t])/model.C_B
            return model.soc[t] == model.soc[t-1] + model.Delta[t]*(model.i_p[t] - model.i_n[t])/model.C_B

        def soc_lim_1_rule(model, t):
            return model.S_B_lower <= model.soc[t]

        def soc_lim_2_rule(model, t):
            return model.soc[t] <= model.S_B_upper

        def power_lim_n_1_rule(model, t):
            return model.P_B_lower*model.ybd[t] <= model.wbd[t]

        def power_lim_n_2_rule(model, t):
            return model.wbd[t] <= model.P_B_upper*model.ybd[t]

        def power_lim_p_1_rule(model, t):
            return model.P_B_lower*model.ybc[t] <= model.wbc_csp[t] + model.wbc_pv[t]

        def power_lim_p_2_rule(model, t):
            return model.wbc_pv[t] + model.wbc_csp[t] <= model.P_B_upper*model.ybc[t]

        def curr_lim_rule(model, t):
            if t == model.t_start:
                return model.i_n[t] <= model.I_upper_n*model.soc0 
            return model.i_n[t] <= model.I_upper_n*model.soc[t-1]

        def gradient_rule(model, t):
            if t == model.t_start:
                return model.i_p[t] <= model.C_B*(1-model.soc0)/model.Delta[t]
            return model.i_p[t] <= model.C_B*(1-model.soc[t-1])/model.Delta[t]

        def curr_lim_n_1_rule(model, t):
            return model.I_lower_n*model.ybd[t] <= model.i_n[t]

        def curr_lim_n_2_rule(model, t):
            return model.i_n[t] <= model.I_upper_n*model.ybd[t]

        def curr_lim_p_1_rule(model, t):
            return model.I_lower_p*model.ybc[t] <= model.i_p[t]

        def curr_lim_p_2_rule(model, t):
            return model.i_p[t] <= model.I_upper_p*model.ybc[t]

        def one_state_rule(model, t):
            return model.ybc[t] + model.ybd[t] <= 1

        def pow_lim_p_sun_rule(model, t):
            return model.wbc_pv[t] + model.wbc_csp[t] == model.A_V*model.z_p[t] + (model.B_V + model.I_avg*model.R_int)*model.x_p[t]

        def pow_lim_n_rule(model, t):
            return model.wbd[t] == model.A_V*model.z_n[t] + (model.B_V - model.I_avg*model.R_int)*model.x_n[t]
 
        self.model.battery_balance_con = pe.Constraint(self.model.T, rule=battery_balance_rule)
        self.model.soc_lim_1_con = pe.Constraint(self.model.T, rule=soc_lim_1_rule)
        self.model.soc_lim_2_con = pe.Constraint(self.model.T, rule=soc_lim_2_rule)
        self.model.power_lim_n_1_con = pe.Constraint(self.model.T, rule=power_lim_n_1_rule)
        self.model.power_lim_n_2_con = pe.Constraint(self.model.T, rule=power_lim_n_2_rule)
        if self.include["pv"]:
            self.model.power_lim_p_1_con = pe.Constraint(self.model.T, rule=power_lim_p_1_rule)
            self.model.power_lim_p_2_con = pe.Constraint(self.model.T, rule=power_lim_p_2_rule)
        self.model.curr_lim_con = pe.Constraint(self.model.T, rule=curr_lim_rule)
        self.model.gradient_con = pe.Constraint(self.model.T, rule=gradient_rule)
        self.model.curr_lim_n_1_con = pe.Constraint(self.model.T, rule=curr_lim_n_1_rule)
        self.model.curr_lim_n_2_con = pe.Constraint(self.model.T, rule=curr_lim_n_2_rule)
        self.model.curr_lim_p_1_con = pe.Constraint(self.model.T, rule=curr_lim_p_1_rule)
        self.model.curr_lim_p_2_con = pe.Constraint(self.model.T, rule=curr_lim_p_2_rule)
        self.model.one_state_con = pe.Constraint(self.model.T, rule=one_state_rule)
        if self.include["pv"]:
            self.model.pow_lim_p_sun_con = pe.Constraint(self.model.T, rule=pow_lim_p_sun_rule)
        self.model.pow_lim_n_con = pe.Constraint(self.model.T, rule=pow_lim_n_rule)
        
    def addAuxiliaryBatteryConstraints(self):
        def aux_lim_n_1_rule(model, t):
            return model.I_lower_n*model.ybd[t] <= model.x_n[t]

        def aux_lim_n_2_rule(model, t):
            return model.x_n[t] <= model.I_upper_n*model.ybd[t]        

        def aux_lim_p_1_rule(model, t):
            return model.I_lower_p*model.ybc[t] <= model.x_p[t]        

        def aux_lim_p_2_rule(model, t):
            return model.x_p[t] <= model.I_upper_p*model.ybc[t]        

        def aux_lim_rule(model, t):
            if t == model.t_start:
                return model.x_n[t] <= model.I_upper_n*model.soc0
            return model.x_n[t] <= model.I_upper_n*model.soc[t-1]  

        def aux_relate_p_1_rule(model, t):
            return -model.I_upper_p*(1-model.ybc[t]) <= model.i_p[t] - model.x_p[t]

        def aux_relate_p_2_rule(model, t):
            return model.i_p[t] - model.x_p[t] <= model.I_upper_p*(1-model.ybc[t]) 

        def aux_relate_n_1_rule(model, t):
            return -model.I_upper_n*(1-model.ybd[t]) <= model.i_n[t] - model.x_n[t] 

        def aux_relate_n_2_rule(model, t):
            return model.i_n[t] - model.x_n[t] <= model.I_upper_n*(1-model.ybd[t])
        
        self.model.aux_lim_n_1_con = pe.Constraint(self.model.T, rule=aux_lim_n_1_rule)
        self.model.aux_lim_n_2_con = pe.Constraint(self.model.T, rule=aux_lim_n_2_rule)
        self.model.aux_lim_p_1_con = pe.Constraint(self.model.T, rule=aux_lim_p_1_rule)
        self.model.aux_lim_p_2_con = pe.Constraint(self.model.T, rule=aux_lim_p_2_rule)
        self.model.aux_lim_con = pe.Constraint(self.model.T, rule=aux_lim_rule)
        self.model.aux_relate_p_1_con = pe.Constraint(self.model.T, rule=aux_relate_p_1_rule)
        self.model.aux_relate_p_2_con = pe.Constraint(self.model.T, rule=aux_relate_p_2_rule)
        self.model.aux_relate_n_1_con = pe.Constraint(self.model.T, rule=aux_relate_n_1_rule)
        self.model.aux_relate_n_2_con = pe.Constraint(self.model.T, rule=aux_relate_n_2_rule)

    def addBatteryLinearizationConstraints(self):
        def cc_1_rule(model, t):
            if t == model.t_start:
                return model.z_p[t] >= model.I_upper_p*model.soc0 + model.S_B_upper*model.i_p[t] - model.S_B_upper*model.I_upper_p
            return model.z_p[t] >= model.I_upper_p*model.soc[t-1] + model.S_B_upper*model.i_p[t] - model.S_B_upper*model.I_upper_p

        def cc_2_rule(model, t):
            if t == model.t_start:
                 return model.z_p[t] >= model.I_lower_p*model.soc0 + model.S_B_lower*model.i_p[t] - model.S_B_lower*model.I_lower_p
            return model.z_p[t] >= model.I_lower_p*model.soc[t-1] + model.S_B_lower*model.i_p[t] - model.S_B_lower*model.I_lower_p

        def cc_3_rule(model, t):
            if t == model.t_start:
                return model.z_p[t] <= model.I_upper_p*model.soc0 + model.S_B_lower*model.i_p[t] - model.S_B_lower*model.I_upper_p
            return model.z_p[t] <= model.I_upper_p*model.soc[t-1] + model.S_B_lower*model.i_p[t] - model.S_B_lower*model.I_upper_p

        def cc_4_rule(model, t):
            if t == model.t_start:
                return model.z_p[t] <= model.I_lower_p*model.soc0 + model.S_B_upper*model.i_p[t] - model.S_B_upper*model.I_lower_p
            return model.z_p[t] <= model.I_lower_p*model.soc[t-1] + model.S_B_upper*model.i_p[t] - model.S_B_upper*model.I_lower_p

        def cc_5_rule(model, t):
            if t == model.t_start:
                return model.z_n[t] >= model.I_upper_n*model.soc0 + model.S_B_upper*model.i_n[t] - model.S_B_upper*model.I_upper_n
            return model.z_n[t] >= model.I_upper_n*model.soc[t-1] + model.S_B_upper*model.i_n[t] - model.S_B_upper*model.I_upper_n

        def cc_6_rule(model, t):
            if t == model.t_start:
                return model.z_n[t] >= model.I_lower_n*model.soc0 + model.S_B_lower*model.i_n[t] - model.S_B_lower*model.I_lower_n
            return model.z_n[t] >= model.I_lower_n*model.soc[t-1] + model.S_B_lower*model.i_n[t] - model.S_B_lower*model.I_lower_n

        def cc_7_rule(model, t):
            if t == model.t_start:
                return model.z_n[t] <= model.I_upper_n*model.soc0 + model.S_B_lower*model.i_n[t] - model.S_B_lower*model.I_upper_n
            return model.z_n[t] <= model.I_upper_n*model.soc[t-1] + model.S_B_lower*model.i_n[t] - model.S_B_lower*model.I_upper_n

        def cc_8_rule(model, t):
            if t == model.t_start:
                return model.z_n[t] <= model.I_lower_n*model.soc0 + model.S_B_upper*model.i_n[t] - model.S_B_upper*model.I_lower_n
            return model.z_n[t] <= model.I_lower_n*model.soc[t-1] + model.S_B_upper*model.i_n[t] - model.S_B_upper*model.I_lower_n
        
        self.model.cc_1_con = pe.Constraint(self.model.T, rule=cc_1_rule)
        self.model.cc_2_con = pe.Constraint(self.model.T, rule=cc_2_rule)
        self.model.cc_3_con = pe.Constraint(self.model.T, rule=cc_3_rule)
        self.model.cc_4_con = pe.Constraint(self.model.T, rule=cc_4_rule)
        self.model.cc_5_con = pe.Constraint(self.model.T, rule=cc_5_rule)
        self.model.cc_6_con = pe.Constraint(self.model.T, rule=cc_6_rule)
        self.model.cc_7_con = pe.Constraint(self.model.T, rule=cc_7_rule)
        self.model.cc_8_con = pe.Constraint(self.model.T, rule=cc_8_rule)

    def addOperatingAssumptions(self):
        self.model.F_thresh = pe.Param(within=pe.Reals, initialize=0.50)

        # never shutdown the power cycle (run in standby at least)
        ## do not relax this in the global solve,
        ## we require a longer problem horizon to make a decision on cycle shutdown
        def ycoff_force_zero_rule(model, t):
            return model.y[t] + model.ycsu[t] + model.ycsb[t] >= 1

        # squeeze receiver outlet temp to max possible (if operating)
        def T_rout_force_upper_rule(model, t):
            return model.T_rout[t] <= model.F[t] * (model.T_rout_max - model.T_cs[t]) + model.T_cs[t] + model.T_rout_max * (1 - model.yr[t] + model.yrsu[t])

        def T_rout_force_lower_rule(model, t):
            return model.T_rout[t] >= model.F[t] * (model.T_rout_max - model.T_cs[t]) + model.T_cs[t] - model.T_rout_max * (1 - model.yr[t] + model.yrsu[t])

        # give a tighter upper bound on mass flow rate to the receiver
        def rec_mdotr_max_rule(model, t):
            if model.Qin[t] == 0:
                return (
                    model.mdot_r_hs[t] + model.mdot_r_cs[t] <= (
                    model.Qin[t] / (model.F[t] * model.Cp * (model.T_rout_max - model.T_cs_max))) +
                    model.mdot_r_max * model.yrsb[t]
                )
            return (model.mdot_r_hs[t] + model.mdot_r_cs[t]) <= model.mdot_r_max * model.yrsb[t]

        # improves linear solution
        def lower_F_rule(model, t):
            return model.yr[t] <= model.F[t] / model.F_thresh


        self.model.ycoff_force_zero_con = pe.Constraint(self.model.T, rule=ycoff_force_zero_rule)
        self.model.T_rout_force_upper_con = pe.Constraint(self.model.T_nl, rule=T_rout_force_upper_rule)
        self.model.T_rout_force_lower_con = pe.Constraint(self.model.T_nl, rule=T_rout_force_lower_rule)
        self.model.rec_mdotr_max_con = pe.Constraint(self.model.T_nl, rule=rec_mdotr_max_rule)
        self.model.lower_F_con = pe.Constraint(self.model.T_l, rule=lower_F_rule)

    def addDiscretizedMdotc(self):
        self.model.n_mdotc = pe.Param(within=pe.NonNegativeIntegers, initialize=4);
        self.model.Kmc = pe.Set(initialize=range(1, self.model.n_mdotc+1))
        delta_mdotc_dict = {}
        for kidx in range(1, self.model.n_mdotc+1):
            #delta_mdotc_dict[k] = self.model.mdot_c_min * (4 - 1.2) / self.model.n_mdotc   Bug in code - fix below
            delta_mdotc_dict[kidx] = self.model.mdot_c_min * (1.2 + 2.8*(kidx-1)/3) / self.model.n_mdotc  # Kmc equal steps from 1.2*min to 4*min (same as max)
        self.model.k_mdotc = pe.Param(self.model.Kmc, within=pe.Reals, initialize=delta_mdotc_dict)
        self.model.mdot_c_min_su = pe.Param(within=pe.Reals, initialize=(self.model.Qc/(self.model.Cp*(5 - self.model.T_cout_su))))
        self.model.T_cout_su = pe.Param(within=pe.Reals, initialize=2.90)
        self.model.theta_mdotc = pe.Var(self.model.T_nl * self.model.Kmc, domain=pe.Binary)

        def mdotc_theta_cut_rule(model, t, k):
            return model.theta_mdotc[t, k] <= model.y[t]

        def mdotc_theta_prec_rule(model, t, k):
            return model.theta_mdotc[t, k] >= model.theta_mdotc[t, k+1]

        def force_mdotc_rule(model, t):
            return model.mdot_c[t] == model.mdot_c_min_su*model.ycsu[t] + 1.2*model.mdot_c_min*y[t] + sum(
                 model.delta_mdotc[k] * self.model.theta_mdotc[t, k] for k in model.Kmc)

        self.model._con = pe.Constraint(self.model.T_nl, self.model.Kmc, rule=mdotc_theta_cut_rule)
        self.model.mdotc_theta_prec_con = pe.Constraint(self.model.T_nl, self.model.Kmc, rule=mdotc_theta_prec_rule)
        self.model.force_mdotc_con = pe.Constraint(self.model.T_nl, self.model.Kmc, rule=force_mdotc_rule)

    def addGridSignalConstraints(self):
        def overprod_signal_rule(model, h):
            return model.g_plus[h] >= sum(model.Delta[t] * model.wdot_s[t] for t in model.T_h[h]) - (model.G[h] + model.day_ahead_tol_plus)

        def underprod_signal_rule(model, h):
            return model.g_minus[h] >= (model.G[h] - model.day_ahead_tol_minus) - sum(model.Delta[t] * model.wdot_s[t] for t in model.T_h[h])

        self.model.overprod_signal_con = pe.Constraint(self.model.H, rule=overprod_signal_rule)
        self.model.underprod_signal_con = pe.Constraint(self.model.H, rule=underprod_signal_rule)

    def generateConstraints(self):
        if self.include["persistence"]:
            self.addPersistenceConstraints()
        # self.addReceiverStartupConstraintsLinear()
        if self.include["simple_receiver"]:
            self.addSimpleReceiverConstraint()
        else:
            self.addReceiverStartupConstraints()
            self.addReceiverSupplyAndDemandConstraints()
            self.addReceiverShutdownConstraints()
            self.addReceiverPenaltyConstraints()
            self.addReceiverModeLogicConstraints()
            self.addReceiverMassFlowRateConstraints()
            self.addReceiverTemperatureConstraints()
            self.addReceiverPowerBalanceConstraints()
        self.addTESEnergyBalanceConstraints()
        self.addThermalStorageMassTempConstraints()
        self.addCycleStartupConstraints()
        self.addPowerCycleThermalInputConstraints()
        self.addPowerCycleMassFlowRateConstraints()
        self.addPowerCycleTemperatureConstraints()
        self.addPowerCycleEnergyOutputConstraints()
        self.addPowerCycleOutputRampingConstraints()
        self.addElectricBalanceConstraints()
        self.addMinUpAndDowntimeConstraints()
        self.addPowerCyclePenaltyConstraints()
        self.addCycleLogicConstraints()
        if self.include["pv"]:
            self.addPVConstraints()
        if self.include["battery"]:
            self.addBatteryConstraints()
            self.addAuxiliaryBatteryConstraints()
            self.addBatteryLinearizationConstraints()
        if self.include["op_assumptions"]:
            self.addOperatingAssumptions()
        if self.include["signal"]:
            self.addGridSignalConstraints()

    def solveModel(self, mipgap=0.001, solver='cbc', timelimit=60, tee=False, keepfiles=False):
        if solver == 'cbc':
            opt = pe.SolverFactory('cbc')
            opt.options["ratioGap"] = mipgap
            opt.options["seconds"] = timelimit
        elif solver == 'cplex':
            opt = pe.SolverFactory('cplex')
            opt.options["mipgap"] = mipgap
            opt.options["timelimit"] = timelimit
        elif solver != 'ipopt':
            raise (ValueError("solver %s not supported" % solver))
        else:
            opt = pe.SolverFactory('ipopt')
        results = opt.solve(self.model, tee=tee, keepfiles=keepfiles)
        return results
    
    def printCycleOutput(self):
        for t in self.model.T:
            if self.model.ycge[t].value > 1e-3:
                print("Cycle off at period ", t, " - Time = ", self.model.Delta_e[t])
            if self.model.ycgb[t].value > 1e-3:
                print("Cycle on at period ", t, " - Time = ", self.model.Delta_e[t])
                
    def populate_variable_values(self, m):
        """
        Populate model variable values using a previously solved model as input.  Used in multi-phased solution
        approach.

        Parameters
        =============
        m : pyomo.ConcreteModel | model containing variable values

        Returns
        =============
        None (populates values within self.model)
        """
        #send outputs to model instance
        d={}
        for v1 in m.model.component_objects(pe.Var, active=True):
            try:
                d[v1.name] = {idx : pe.value(v1[idx]) for idx in v1}
            except ValueError:
                print("warning: no values for variable ", v1.name)
                pass
        for v2 in self.model.component_objects(pe.Var, active=True):  #TODO find a better way than a double loop
            if v2.name in d.keys() and len(list(d[v2.name].keys())) > 0:
                for i in v2:  #only populate relevant values in new model; nonlinear values handled separately
                    v2[i].set_value(d[v2.name][i])
                continue
        return

    def fix_binaries(self):
        """
        fixes all binary variables.
        """
        for v in self.model.component_objects(pe.Var, active=True):
            for i in v:   #note that a scalar parameter's indices in pyomo are [None] so this does apply
                if v[i].domain == pe.Binary:
                    v[i].fix()
        return


    
# if __name__ == "__main__": 
#     import dispatch_params
#     import dispatch_outputs
#     params = dispatch_params.buildParamsFromAMPLFile("./input_files/data_energy.dat")
#     params.start = 1        # as borrowed from run_phase_one.py
#     params.stop = 68        # as borrowed from run_phase_one.py
#     params.transition = 0   # as the default in run_dispatch() signature and as hardcoded in run_phase_one()
#     include = {"pv": False, "battery": False, "persistence": False, "force_cycle": True}
#     rt = RealTimeDispatchModel(params, include)
#     rt_results = rt.solveModel()
#     outputs = dispatch_outputs.RTDispatchOutputs(rt.model)
#     outputs.print_outputs()
    
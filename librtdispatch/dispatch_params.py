# -*- coding: utf-8 -*-
"""
Data structures for the real-time dispatch model
"""

import pyomo.environ as pe
#import re

def buildParamsFromFiles(filenames):
    params_dict = {}
    for fname in filenames:
        # print(fname)
        params_dict = buildParamsFromAMPLFile(fname, params_dict)
    params_dict = adjustParamsDict(params_dict)
    return params_dict

def buildParamsFromAMPLFile(filename, params_dict={}):
    f = open(filename, 'r')
    lines = f.readlines()
    in_param = False
    # lineno = 1
    for line in lines:
        # print(lineno)
        # lineno +=1
        if line.startswith("let"):
            in_param = False
            splitline = line.split(" ")
            key = splitline[1]
            if key in ["T", "day_of_year", "nc", "nfw", "transition"]:
                val = int(splitline[-1][:-2])
            else:
                val = float(splitline[-1][:-2])
            params_dict[key] = val
            in_param = False
        elif line.startswith("param "):  #If parameter header line, start
            in_param = True
            splitline = line.split(" ")
            param_key = splitline[1]
            params_dict[param_key] = {}
        elif in_param:
            if line[0] == ";":
                in_param = False
                param_key = None
            else: 
                splitline = line.split("\t")
                params_dict[param_key][int(splitline[0])] = float(splitline[-1])
    return params_dict

def adjustParamsDict(params_dict):
    ##Adjustments for john's scripts
    params_dict["avg_price"] = 0.0
    sum_delta = 0.0
    params_dict["W_u_plus"] = {}
    params_dict["W_u_minus"] = {}
    params_dict["Wdotnet"] = {}
    params_dict["F"] = {}
    params_dict["P_field_rec"] = {}
    for t in params_dict["Delta"].keys():
        params_dict["W_u_plus"][t] = params_dict["Wdotl"] + params_dict["W_delta_plus"] * params_dict["Delta"][t] / 2
        params_dict["W_u_minus"][t] = params_dict["Wdotl"] + params_dict["W_delta_minus"] * params_dict["Delta"][t] / 2
        params_dict["Wdotnet"][t] = params_dict["Wdotu"]
        params_dict["avg_price"] += params_dict["Delta"][t] * params_dict["P"][t]
        sum_delta += params_dict["Delta"][t]
        params_dict["F"][t] = int(params_dict["Qin"][t] > 0)
        params_dict["P_field_rec"][t] = params_dict["Qin"][t] * 0.01
    params_dict["avg_price"] /= sum_delta
    params_dict["G"] = {}
    params_dict["Cg_plus"] = {}
    params_dict["Cg_minus"] = {}
    params_dict["num_signal_hours"] = 24
    for h in range(1,25):
        params_dict["Cg_plus"][h] = 10
        params_dict["Cg_minus"][h] = 10
        if h in [10,20]:
            params_dict["G"][h] = 45000
        elif h in [11,12,13,14,15,16,17,18,19]:
            params_dict["G"][h] = 100000
        else:
            params_dict["G"][h] = 0
    params_dict["T_cs_des"] = 295
    params_dict["T_hs_des"] = 565
    # params_dict["avg_price"] = 0.138
    params_dict["avg_purchase_price"] = 0.03
    params_dict["day_ahead_tol_plus"] = 5000
    params_dict["day_ahead_tol_minus"] = 5000
    params_dict["avg_price_disp_storage_incentive"] = 0.0
    return params_dict

def getDispatchParamsFromDict(params_dict):
    """
    Translates dictionary into parameter object.
    :param params_dict: dictionary containing key-val pairs in which keys are parameter names, and vals are either
        scalars or linked lists representing time-series inputs.
    :return params: DispatchParams object to be used as input to SSC and dispatch model
    """
    import dispatch
    params = dispatch.DispatchParams()
    # ---------------------------------------------------------------------
    # Indexing
    params.T = params_dict.get("T")
    params.start = params_dict.get("start")
    params.stop = params_dict.get("stop")
    params.transition = params_dict.get("transition")
    params.nc = params_dict.get("nc")
    params.nfw = params_dict.get("nfw")
    params.num_signal_hours = params_dict.get("num_signal_hours")

    # ---------------------------------------------------------------------
    # Piecewise-linear indexed parameters
    params.Pc = params_dict.get("Pc")
    params.Bc = params_dict.get("Bc")
    params.Pfw = params_dict.get("Pfw")
    params.Bfw = params_dict.get("Bfw")

    # --------------------------------------------------------------------
    # Field and receiver parameters
    params.Drsu = params_dict.get("Drsu")  # Minimum time to start the receiver (hr)
    params.Drsd = params_dict.get("Drsd")  # Minimum time to shut down the receiver (hr)
    params.Er = params_dict.get("Er")  # Required energy expended to start receiver (kWht)
    params.Qrl = params_dict.get("Qrl")  # Minimum operational thermal power delivered by receiver (kWt)
    params.Qrsb = params_dict.get("Qrsb")  # Required thermal power for receiver standby (kWt)
    params.Qrsd = params_dict.get("Qrsd")  # Required thermal power for receiver shut down (kWht?)
    params.Qru = params_dict.get("Qru")  # Allowable power per period for receiver start-up (kWt)
    params.mdot_r_min = params_dict.get("mdot_r_min")  # Minimum mass flow rate of HTF to the receiver (kg/s)
    params.mdot_r_max = params_dict.get("mdot_r_max")  # Maximum mass flow rate of HTF to the receiver (kg/s)
    params.T_rout_min = 0.#params_dict.get("T_rout_min")  # Minimum allowable receiver outlet T (C)
    params.T_rout_max = params_dict.get("T_rout_max")  # Maximum allowable receiver outlet T (C)

    # --------------------------------------------------------------------
    # TES parameters
    params.Eu = params_dict.get("Eu")  # Thermal energy storage capacity (kWht)
    params.Cp = params_dict.get("Cp")  # Specific heat of HTF (kJ/kg/K)
    params.mass_cs_min = params_dict.get("mass_cs_min")  # Minimum mass of HTF in cold storage (kg)
    params.mass_cs_max = params_dict.get("mass_cs_max")  # Maximum mass of HTF in cold storage (kg)
    params.mass_hs_min = params_dict.get("mass_hs_min")  # Minimum mass of HTF in hot storage (kg)
    params.mass_hs_max = params_dict.get("mass_hs_max")  # Maximum mass of HTF in hot storage (kg)
    params.T_cs_min = params_dict.get("T_cs_min")  # Minimum HTF T in cold storage (C)
    params.T_cs_max = params_dict.get("T_cs_max")  # Maximum HTF T in cold storage (C)
    params.T_cs_des = params_dict.get("T_cs_des")  # Design point HTF T in cold storage (C)
    params.T_hs_min = params_dict.get("T_hs_min")  # Minimum HTF T in cold storage (C)
    params.T_hs_max = params_dict.get("T_hs_max")  # Maximum HTF T in hot storage (C)
    params.T_hs_des = params_dict.get("T_hs_des")  # Design point HTF T in hot storage (C)

    # --------------------------------------------------------------------
    # Cycle parameters
    params.Ec = params_dict.get("Ec")  # Required energy expended to start cycle (kWht)
    params.Ew = params_dict.get("Ew")  # Required energy expended to warm-start the cycle (kWht)
    params.eta_des = params_dict.get("eta_des")  # Cycle design point efficiency
    params.etap = params_dict.get("etap")  # Slope of linear approximation to power cycle performance curve (kWe/kWt)
    params.Qb = 0.0# params_dict.get("Qb")  # Cycle standby thermal power consumption (kWt)
    params.Qc = params_dict.get("Qc")  # Allowable power per period for cycle startup (kWt)
    params.Ql = params_dict.get("Ql")  # Minimum operational thermal power input to cycle (kWt)
    params.Qu = params_dict.get("Qu")  # Cycle thermal power capacity (kWt)
    params.kl = params_dict.get("kl")  # Change in lower bound of cycle thermal load due to hot storage temperature (kWt/C)
    params.ku = params_dict.get("ku")  # Change in upper bound of cycle thermal load due to hot storage temperature (kWt/C)
    params.Wdot_design = params_dict.get("Wdot_design")  # Design point electrical output of the power cycle (kWe)
    params.Wdot_p_max = params_dict.get("Wdot_p_max")
    params.Wdotl = params_dict.get("Wdotl")  # Minimum cycle electric power output (kWe)
    params.Wdotu = params_dict.get("Wdotu")  # Cycle electric power rated capacity (kWe)
    params.delta_T_design = params_dict.get("delta_T_design")  # Design point temperature change of HTF across the SGS model (C)
    params.delta_T_max = params_dict.get("delta_T_max")  # Max temperature change of HTF across the SGS model (C)
    params.mdot_c_design = params_dict.get("mdot_c_design")  # Design point mass flow rate of HTF to the power cycle (kg/s)
    params.mdot_c_min = params_dict.get("mdot_c_min")  # Minmium mass flow rate of HTF to the power cycle (kg/s)
    params.mdot_c_max = params_dict.get("mdot_c_max")  # Maximum mass flow rate of HTF to  the power cycle (kg/s)
    params.T_cin_design = params_dict.get("T_cin_design")  # HTF design point power cycle inlet temperature (C)
    params.T_cout_min = 0.0  # params_dict.get("T_cout_min")  # HTF design point power cycle inlet temperature (C)
    params.T_cout_max = 0.0  # params_dict.get("T_cout_max")  # HTF design point power cycle inlet temperature (C)
    params.W_delta_plus = params_dict.get("W_delta_plus")  # Power cycle ramp-up designed limit (kWe/h)
    params.W_delta_minus = params_dict.get("W_delta_minus")
    params.W_v_plus = params_dict.get("W_v_plus")
    params.W_v_minus = params_dict.get("W_v_minus")
    params.Yu = params_dict.get("Yu")
    params.Yd = params_dict.get("Yd")

    # --------------------------------------------------------------------
    # Parastic loads
    params.Ehs = params_dict.get("Ehs")
    params.Wh_track = params_dict.get("Wh_track")
    params.Wh_comm = params_dict.get("Wh_comm")
    params.Lr = params_dict.get("Lr")
    params.Pr = params_dict.get("Pr")
    params.Wht_full = params_dict.get("Wht_full")
    params.Wht_part = params_dict.get("Wht_part")
    params.Lc = params_dict.get("Lc")
    params.Wb = params_dict.get("Wb")
    params.Wc = params_dict.get("Wc")

    # --------------------------------------------------------------------
    # Cost parameters
    params.alpha = params_dict.get("alpha")
    params.Crec = params_dict.get("Crec")
    params.Crsu = params_dict.get("Crsu")
    params.Crhsp = params_dict.get("Crhsp")
    params.Cpc = params_dict.get("Cpc")
    params.Ccsu = params_dict.get("Ccsu")
    params.Cchsp = params_dict.get("Cchsp")
    params.C_delta_w = params_dict.get("C_delta_w")
    params.C_v_w = params_dict.get("C_v_w")
    params.Ccsb = params_dict.get("Ccsb")

    # --------------------------------------------------------------------
    # Grid parameters
    params.G = params_dict.get("G")
    params.Cg_plus = params_dict.get("Cg_plus")
    params.Cg_minus = params_dict.get("Cg_minus")
    params.day_ahead_tol_plus = params_dict.get("day_ahead_tol_plus")
    params.day_ahead_tol_minus = params_dict.get("day_ahead_tol_minus")

    # --------------------------------------------------------------------
    # Regression parameters
    params.alpha_b = params_dict.get("alpha_b")
    params.alpha_T = params_dict.get("alpha_T")
    params.alpha_m = params_dict.get("alpha_m")
    params.beta_b = params_dict.get("beta_b")
    params.beta_T = params_dict.get("beta_T")
    params.beta_m = params_dict.get("beta_m")
    params.beta_mT = params_dict.get("beta_mT")

    # ---------------------------------------------------------------------
    # Initial conditions
    params.s0 = 0.0  # params_dict.get("s0")
    params.ucsu0 = params_dict.get("ucsu0")
    params.ursu0 = params_dict.get("ursu0")
    params.ursd0 = params_dict.get("ursd0")
    params.wdot0 = params_dict.get("wdot0")
    params.mass_cs0 = params_dict.get("mass_cs0")
    params.mass_hs0 = params_dict.get("mass_hs0")
    params.T_cs0 = params_dict.get("T_cs0")
    params.T_hs0 = params_dict.get("T_hs0")

    params.yr0 = params_dict.get("yr0")
    params.yrsb0 = params_dict.get("yrsb0")
    params.yrsd0 = params_dict.get("yrsd0")
    params.yrsu0 = params_dict.get("yrsu0")

    params.y0 = params_dict.get("y0")
    params.ycsb0 = params_dict.get("ycsb0")
    params.ycsu0 = params_dict.get("ycsu0")

    params.drsd0 = params_dict.get("drsd0")
    params.drsu0 = params_dict.get("drsu0")
    params.Yu0 = params_dict.get("Yu0")
    params.Yd0 = params_dict.get("Yd0")

    # ---------------------------------------------------------------------
    # Time-indexed parameters
    params.Delta = params_dict.get("Delta")
    params.Delta_e = params_dict.get("Delta_e")
    params.D = params_dict.get("D")
    params.P = params_dict.get("P")
    params.Wdotnet = params_dict.get("Wdotnet")
    params.W_u_plus = params_dict.get("W_u_plus")
    params.W_u_minus = params_dict.get("W_u_minus")

    # Time-indexed parameters derived from ssc estimates
    params.Qin = params_dict.get("Qin")
    params.delta_rs = params_dict.get("delta_rs")
    params.delta_cs = params_dict.get("delta_cs")
    params.etaamb = params_dict.get("etaamb")
    params.etac = params_dict.get("etac")
    params.F = params_dict.get("F")
    params.Q_cls = params_dict.get("Q_cls")

    # Added parameters for optimization model taken or derived from SSC
    params.day_of_year = params_dict.get("day_of_year")
    params.disp_time_weighting = params_dict.get("disp_time_weighting")
    params.avg_price = params_dict.get("avg_price")
    params.avg_purchase_price = params_dict.get("avg_purchase_price")
    params.avg_price_disp_storage_incentive = params_dict.get("avg_price_disp_storage_incentive")

    # Provided parasitic loads and thermal output to receiver when using simplified model
    params.P_field_rec = params_dict.get("P_field_rec")

    return params


if __name__ == "__main__":
    filename = "./input_files/data_energy.dat"
    params_dict = buildParamsFromAMPLFile(filename)
    print("T: ",params_dict["T"])
    print("eff_frac: \n",params_dict["eff_frac"])
            
            
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
    for t in params_dict["Delta"].keys():
        params_dict["W_u_plus"][t] = params_dict["Wdotl"] + params_dict["W_delta_plus"] * params_dict["Delta"][t] / 2
        params_dict["W_u_minus"][t] = params_dict["Wdotl"] + params_dict["W_delta_minus"] * params_dict["Delta"][t] / 2
        params_dict["Wdotnet"][t] = params_dict["Wdotu"]
        params_dict["avg_price"] += params_dict["Delta"][t] * params_dict["P"][t]
        sum_delta += params_dict["Delta"][t]
        params_dict["F"][t] = int(params_dict["Qin"][t] > 0)
    params_dict["avg_price"] /= sum_delta
    return params_dict


if __name__ == "__main__":
    filename = "./input_files/data_energy.dat"
    params_dict = buildParamsFromAMPLFile(filename)
    print("T: ",params_dict["T"])
    print("eff_frac: \n",params_dict["eff_frac"])
            
            
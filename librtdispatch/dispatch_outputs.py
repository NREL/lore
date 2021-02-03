# -*- coding: utf-8 -*-
"""
Real-time dispatch optimization model output processing module
"""
import pyomo.environ as pe
import numpy


class SimpleDispatchOutputs(object):
    def __init__(self, model):
        """
        only records receiver thermal power and power cycle outputs.
        :param model:
        """
        self.objective_value = pe.value(model.OBJ)
        self.cycle_on = numpy.array([pe.value(model.y[t]) for t in model.T])
        self.cycle_standby = numpy.array([pe.value(model.ycsb[t]) for t in model.T])
        self.cycle_startup = numpy.array([pe.value(model.ycsu[t]) for t in model.T])
        self.receiver_power = numpy.array([pe.value(model.xr[t]) for t in model.T])
        self.thermal_input_to_cycle = numpy.zeros_like(self.receiver_power)
        for t in model.T:
            if t in model.T_nl:
                self.thermal_input_to_cycle[t-model.t_start] = pe.value(model.x_calc[t])
            else:
                self.thermal_input_to_cycle[t-model.t_start] = pe.value(model.x[t])
        self.electrical_output_from_cycle = numpy.array([pe.value(model.wdot[t]) for t in model.T])
        self.net_electrical_output = numpy.array([pe.value(model.wdot_s[t]) for t in model.T])
        self.tes_soc = numpy.array([pe.value(model.s[t]) for t in model.T])

    def print_outputs(self):
        print("objective_value: ")
        print(self.objective_value)
        print("cycle_on: ")
        print(self.cycle_on)
        print("cycle_standby: ")
        print(self.cycle_standby)
        print("cycle_startup: ")
        print(self.cycle_startup)
        print("receiver_power: ")
        print(self.receiver_power)
        print("thermal_input_to_cycle: ")
        print(self.thermal_input_to_cycle)
        print("electrical_output_from_cycle: ")
        print(self.electrical_output_from_cycle)
        print("tes_soc: ")
        print(self.tes_soc)

class RTDispatchOutputs(object):
    def __init__(self, model):
        self.objective_value = pe.value(model.OBJ)
        self.cycle_on = numpy.array([pe.value(model.y[t]) for t in model.T])
        self.cycle_standby = numpy.array([pe.value(model.ycsb[t]) for t in model.T])
        self.cycle_startup = numpy.array([pe.value(model.ycsu[t]) for t in model.T])
        self.receiver_on = numpy.array([pe.value(model.yr[t]) for t in model.T])
        self.receiver_startup = numpy.array([pe.value(model.yrsu[t]) for t in model.T])
        self.receiver_standby = numpy.array([pe.value(model.yrsb[t]) for t in model.T])
        self.receiver_power = numpy.array([pe.value(model.xr[t]) for t in model.T])
        self.thermal_input_to_cycle = numpy.zeros_like(self.receiver_power)
        for t in model.T:
            if t in model.T_nl:
                self.thermal_input_to_cycle[t-model.t_start] = pe.value(model.x_calc[t])
            else:
                self.thermal_input_to_cycle[t-model.t_start] = pe.value(model.x[t])
        self.electrical_output_from_cycle = numpy.array([pe.value(model.wdot[t]) for t in model.T])
        self.net_electrical_output = numpy.array([pe.value(model.wdot_s[t]) for t in model.T])
        self.tes_soc = numpy.array([pe.value(model.s[t]) for t in model.T])


        #track num_nl_periods
        self.num_nl_periods = model.t_transition - model.t_start + 1
        self.t_start = model.t_start
        #Additional outputs from optimization model (note some repeat from above)
        self.s0 = pe.value(model.s0)
        #Continuous
        self.drsu = numpy.array([pe.value(model.drsu[t]) for t in model.T])
        self.drsd = numpy.array([pe.value(model.drsd[t]) for t in model.T])
        self.frsd = numpy.array([pe.value(model.frsd[t]) for t in model.T])
        self.frsu = numpy.array([pe.value(model.frsu[t]) for t in model.T])
        self.lr = numpy.array([pe.value(model.lr[t]) for t in model.T_nl])
        self.lc = numpy.array([pe.value(model.lc[t]) for t in model.T_nl])
        self.lfw = numpy.array([pe.value(model.lfw[t]) for t in model.T_nl])
        self.mass_cs = numpy.array([pe.value(model.mass_cs[t]) for t in model.T_nl])
        self.mass_hs = numpy.array([pe.value(model.mass_hs[t]) for t in model.T_nl])
        self.mdot_c = numpy.array([pe.value(model.mdot_c[t]) for t in model.T_nl])
        self.mdot_r_cs = numpy.array([pe.value(model.mdot_r_cs[t]) for t in model.T_nl])
        self.mdot_r_hs= numpy.array([pe.value(model.mdot_r_hs[t]) for t in model.T_nl])
        self.s = numpy.array([pe.value(model.s[t]) for t in model.T_l])
        self.T_cout = numpy.array([pe.value(model.T_cout[t]) for t in model.T_nl])
        self.T_cs = numpy.array([pe.value(model.T_cs[t]) for t in model.T_nl])
        self.T_hs = numpy.array([pe.value(model.T_hs[t]) for t in model.T_nl])
        self.T_rout = numpy.array([pe.value(model.T_rout[t]) for t in model.T_nl])
        self.ucsu = numpy.array([pe.value(model.ucsu[t]) for t in model.T])
        #self.ucsd = numpy.array([pe.value(model.ucsd[t]) for t in model.T])
        self.ursu = numpy.array([pe.value(model.ursu[t]) for t in model.T])
        self.ursd = numpy.array([pe.value(model.ursd[t]) for t in model.T])
        self.wdot = numpy.array([pe.value(model.wdot[t]) for t in model.T])
        self.wdot_delta_plus = numpy.array([pe.value(model.wdot_delta_plus[t]) for t in model.T])
        self.wdot_delta_minus = numpy.array([pe.value(model.wdot_delta_minus[t]) for t in model.T])
        self.wdot_v_plus = numpy.array([pe.value(model.wdot_v_plus[t]) for t in model.T])
        self.wdot_v_minus = numpy.array([pe.value(model.wdot_v_minus[t]) for t in model.T])
        self.wdot_s = numpy.array([pe.value(model.wdot_s[t]) for t in model.T])
        self.wdot_p = numpy.array([pe.value(model.wdot_p[t]) for t in model.T])
        self.x = numpy.array([pe.value(model.x[t]) for t in model.T_l])
        self.xr = numpy.array([pe.value(model.xr[t]) for t in model.T])
        #self.xrsu = numpy.array([pe.value(model.xrsu[t]) for t in model.T])
        #Binary
        self.yr = numpy.array([pe.value(model.yr[t]) for t in model.T])
        self.yrhsp = numpy.array([pe.value(model.yrhsp[t]) for t in model.T])
        self.yrsb = numpy.array([pe.value(model.yrsb[t]) for t in model.T])
        self.yrsd = numpy.array([pe.value(model.yrsd[t]) for t in model.T])
        self.yrsdp = numpy.array([pe.value(model.yrsdp[t]) for t in model.T])
        self.yrsu = numpy.array([pe.value(model.yrsu[t]) for t in model.T])
        self.yrsup = numpy.array([pe.value(model.yrsup[t]) for t in model.T])
        self.y = numpy.array([pe.value(model.y[t]) for t in model.T])
        self.ychsp = numpy.array([pe.value(model.ychsp[t]) for t in model.T])
        self.ycsb = numpy.array([pe.value(model.ycsb[t]) for t in model.T])
        self.ycsd = numpy.array([pe.value(model.ycsd[t]) for t in model.T])
        self.ycsdp = numpy.array([pe.value(model.ycsdp[t]) for t in model.T])
        self.ycsu = numpy.array([pe.value(model.ycsu[t]) for t in model.T])
        self.ycsup = numpy.array([pe.value(model.ycsup[t]) for t in model.T])
        self.ycgb = numpy.array([pe.value(model.ycgb[t]) for t in model.T])
        self.ycge = numpy.array([pe.value(model.ycge[t]) for t in model.T])

        
    def print_outputs(self):
        print("objective_value: ")
        print(self.objective_value)
        print("cycle_on: ")
        print(self.cycle_on)
        print("cycle_standby: ")
        print(self.cycle_standby)
        print("cycle_startup: ")
        print(self.cycle_startup)
        print("receiver_power: ")
        print(self.receiver_power)
        print("receiver_on: ")
        print(self.receiver_on)
        print("receiver_startup: ")
        print(self.receiver_startup)
        print("receiver_standby: ")
        print(self.receiver_standby)
        print("thermal_input_to_cycle: ")
        print(self.thermal_input_to_cycle)
        print("electrical_output_from_cycle: ")
        print(self.electrical_output_from_cycle)
        print("tes_soc: ")
        print(self.tes_soc)

    def sendOutputsToFile(self,fname="dispatch_outputs.csv"):
        outfile = open(fname, 'w')
        outfile.write("s0,"+str(self.s0)+"\n\n")
        outfile.write("t,drsu,drsd,frsd,frsu,s,lr,lc,lfw,mass_cs,mass_hs,mdot_c,mdot_r_cs,m_dot_hs,"+
                      "T_cout,T_cs,T_hs,T_rout,ucsu,ursu,ursd,wdot,wdot_delta_plus,w_dot_delta_minus,"+
                      "wdot_v_plus,wdot_v_minus,wdot_s,wdot_p,x,xr,yr,yrhsp,yrsb,yrsd,yrsdp,yrsu,yrsup,y,ychsp,"+
                      "ycsb,ycsd,ycsdp,ycsu,ycsup,ycgb,ygce"+ #"ycoff"+
                      "\n")
        for t in range(len(self.y)):
            outfile.write(str(t+self.t_start)+","+str(self.drsu[t])+","+str(self.drsd[t])+","+str(self.frsd[t])+","+
                          str(self.frsu[t])+",")
            if t >= self.num_nl_periods:
                outfile.write(str(self.s[t]))
            outfile.write(",")
            if t < self.num_nl_periods:
                outfile.write(str(self.lr[t])+","+str(self.lc[t])+","+str(self.lfw[t])+","+str(self.mass_cs[t])+","+
                              str(self.mass_hs[t])+","+str(self.mdot_c[t])+","+str(self.mdot_r_cs[t])+","+
                              str(self.mdot_r_hs[t])+","+str(self.T_cout[t])+","+str(self.T_cs[t])+","+
                              str(self.T_hs[t])+","+str(self.T_rout[t])+",")
            else:
                outfile.write(",,,,,,,,,,,,")
            outfile.write(str(self.ucsu[t])+","+","+str(self.ursu[t])+","+
                          str(self.ursd[t])+","+str(self.wdot[t])+","+str(self.wdot_delta_plus[t])+","+
                          str(self.wdot_delta_minus[t])+","+str(self.wdot_v_plus[t])+","+str(self.wdot_v_minus[t])+
                          ","+str(self.wdot_s[t])+","+str(self.wdot_p[t])+","+str(self.x[t])+","+str(self.xr[t])+
                          ","+str(self.yr[t])+","+str(self.yrhsp[t])+","+str(self.yrsb[t])+
                          ","+str(self.yrsd[t])+","+str(self.yrsdp[t])+","+str(self.yrsu[t])+","+str(self.yrsup[t])+
                          "," + str(self.y[t]) + "," + str(self.ychsp[t]) + "," + str(self.ycsb[t]) + ","
                          + str(self.ycsd[t]) + "," + str(self.ycsdp[t]) + "," + str(self.ycsu[t]) + ","
                          + str(self.ycsup[t]) + "," + str(self.ycgb[t]) #+ "," + str(self.ycoff[t])
                          + "\n"
                          )
        outfile.close()
"""
runs phase one of John's solution method; i.e., the linear model with a discretization of the mdotc variable.
"""
from pyomo.opt import TerminationCondition
import dispatch_model
import dispatch_params
import dispatch_outputs

# def run_phase_one(input_filenames, include, start, stop, printout = False):
#     params_dict = dispatch_params.buildParamsFromFiles(input_filenames)
#     params_dict["start"] = start
#     params_dict["stop"] = stop
#     params_dict["transition"] = 0
#     if include["persistence"]:
#         assert "wdot_s_prev" in params_dict and "wdot_s_pen" in params_dict
#     params = dispatch_params.getDispatchParamsFromDict(params_dict)
#     rt = dispatch_model.RealTimeDispatchModel(params, include)
#     rt_results = rt.solveModel()
#     if include["simple_receiver"]:
#         outputs = dispatch_outputs.SimpleDispatchOutputs(rt.model)
#         if printout:
#             outputs.print_outputs()
#     else:
#         outputs = dispatch_outputs.RTDispatchOutputs(rt.model)
#         if printout:
#             outputs.print_outputs()
#         outputs.sendOutputsToFile()


# def run_dispatch(params, include, start, stop, transition=0):
#     params.start = start
#     params.stop = stop
#     params.transition = transition
#     rt = dispatch_model.RealTimeDispatchModel(params, include)
#     rt_results = rt.solveModel()
    
#     if rt_results.solver.termination_condition == TerminationCondition.infeasible:
#         return False

#     if include["simple_receiver"]:
#         outputs = dispatch_outputs.SimpleDispatchOutputs(rt.model)
#     else:
#         outputs = dispatch_outputs.RTDispatchOutputs(rt.model)
#     #outputs.print_outputs()
#     #outputs.sendOutputsToFile()
#     return outputs

if __name__ == "__main__":
    input_filenames = [
        "./input_files/_plant_params.dat",
        "./input_files/init_cycle_on_full_tank.dat",
        "./input_files/time_baseline.dat"
    ]
    include = {"pv": False, "battery": False, "persistence": True, "force_cycle": False, "op_assumptions": False,
               "signal": True, "simple_receiver": False}
    start = 1
    stop = 68
    run_phase_one(input_filenames, include, start, stop, True)
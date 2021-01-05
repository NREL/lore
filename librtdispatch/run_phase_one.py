"""
runs phase one of John's solution method; i.e., the linear model with a discretization of the mdotc variable.
"""

import dispatch_model
import dispatch_params
import dispatch_outputs

def run_phase_one(input_filenames, include, start, stop):
    params = dispatch_params.buildParamsFromFiles(input_filenames)
    params["start"] = start
    params["stop"] = stop
    params["transition"] = 0
    rt = dispatch_model.RealTimeDispatchModel(params, include)
    rt_results = rt.solveModel()
    outputs = dispatch_outputs.RTDispatchOutputs(rt.model)
    outputs.print_outputs()

if __name__ == "__main__":
    input_filenames = [
        "./input_files/_plant_params.dat",
        "./input_files/init_cycle_on_full_tank.dat",
        "./input_files/time_baseline.dat"
    ]
    include = {"pv": False, "battery": False, "persistence": False, "force_cycle": False, "op_assumptions": True}
    start = 1
    stop = 68
    run_phase_one(input_filenames, include, start, stop)
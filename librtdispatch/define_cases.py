# -*- coding: utf-8 -*-


def set_inputs_from_name(cs, name):
    
    num, subcase = name

    if num == '1':  # Run field, receiver and cycle from CD data
        cs.control_field = 'CD_data'
        cs.control_receiver = 'CD_data'
        cs.is_optimize = False
        cs.control_cycle = 'CD_data'
        cs.use_CD_measured_reflectivity = True

   
    elif num == '2':    # Run field and receiver from CD data, but optimize cycle dispatch 
        cs.control_field = 'CD_data'
        cs.control_receiver = 'CD_data'
        cs.is_optimize = True
        cs.dispatch_weather_horizon = -1    # Time to switch from actual to forecasted weather (0 = all forecasted, -1 = all actual, >0 = time point (hr))
        cs.use_CD_measured_reflectivity = True
        
            
    elif num == '3':    # Same as case 2, but run with a different relectivity assuming an optimized mirror washing schedule (and correspondingly scaled-up receiver mass flow)
        cs.control_field = 'CD_data'
        cs.control_receiver = 'CD_data'
        cs.is_optimize = True
        cs.dispatch_weather_horizon = -1   
        cs.use_CD_measured_reflectivity = False
        cs.fixed_soiling_loss = 0.02   # 1 - (reflectivity / clean reflectivity)

        
    elif num == '4':    # ssc control of field/receiver with ssc heuristic dispatch
        cs.control_field = 'ssc'
        cs.control_receiver = 'ssc_clearsky'
        cs.is_optimize = False
        cs.control_cycle = 'ssc_heuristic'
        cs.use_CD_measured_reflectivity = True
        
    elif num == '5':  # ssc control of field/receiver with dispatch optimization based on forecasted weather
        cs.control_field = 'ssc'
        cs.control_receiver = 'ssc_clearsky'
        cs.is_optimize = True   
        cs.dispatch_weather_horizon = 0
        cs.use_CD_measured_reflectivity = True

    elif num == '8':  # ssc control of field/receiver with dispatch optimization based on forecasted weather
        cs.control_field = 'ssc'
        cs.control_receiver = 'ssc_clearsky'
        cs.is_optimize = True
        cs.dispatch_weather_horizon = -1
        cs.use_CD_measured_reflectivity = True
        cs.fixed_soiling_loss = 0.02   # 1 - (reflectivity / clean reflectivity)
        
    elif num == '6':  # ssc control of field/receiver with dispatch optimization based on hybrid actual/forecasted weahter
        cs.control_field = 'ssc'
        cs.control_receiver = 'ssc_clearsky'
        cs.is_optimize = True   
        cs.dispatch_weather_horizon = 2  # TODO: what horizon do we want to use?
        cs.use_CD_measured_reflectivity = True       
       
    elif num == '7':  # ssc control of field/receiver with dispatch optimization based on hybrid actual/forecasted weather, and assuming an optimized mirror washing schedule
        cs.control_field = 'ssc'
        cs.control_receiver = 'ssc_clearsky'
        cs.is_optimize = True   
        cs.dispatch_weather_horizon = 2  # TODO: what horizon do we want to use?
        cs.use_CD_measured_reflectivity = False  
        cs.fixed_soiling_loss = 0.02   # 1 - (reflectivity / clean reflectivity)


        

    # Define sub-cases for differnt use of day-ahead schedule
    if subcase == 'a':
        cs.use_day_ahead_schedule = False
    elif subcase == 'b':
        cs.use_day_ahead_schedule = True
        cs.day_ahead_schedule_from = 'calculated'                
    elif subcase == 'c':
        cs.use_day_ahead_schedule = True
        cs.day_ahead_schedule_from = 'NVE'  
        
    return
            

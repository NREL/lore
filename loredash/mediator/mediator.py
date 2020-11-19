from django.conf import settings
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import time
import copy
import datetime
from twisted.internet.task import LoopingCall
from twisted.internet import reactor
import PySAM_DAOTk.TcsmoltenSalt as pysam
from pathlib import Path
from mediator import data_validator, pysam_wrap
# import models

class Mediator:
    pysam_wrap = None
    validated_outputs_prev = None
    update_interval = datetime.timedelta(seconds=5)
    timestep_simulation = datetime.timedelta(minutes=5)

    def __init__(self):
        self.pysam_wrap = pysam_wrap.PysamWrap()
        # Decide whether to preprocess efficiency and flux maps
        if settings.DEBUG == True:
            result = self.pysam_wrap.SetDesign(self.pysam_wrap.design_path)
            if result == 1:
                self.pysam_wrap.PreProcess()
        else:
            self.pysam_wrap.PreProcess()    # do this now so no simulation delay later
    
    def RunOnce(self):
        """For the current point in time, get data from external plant and weather interfaces and run
        entire set of submodels, saving data to database"""

        # The planned code:
        # Step 1:
        #   Thread 1:
        #       a. If virtual plant, query database for needed inputs (or use cache from previous timestep)
        #       b. Call virtual/real plant to get plant operating state and any local weather data
        #       c. Validate these data
        #       d. Store in database and add to current timestep cache
        #   Thread 2:
        #       a. Call module to retrieve weather data(s)
        #       b. Validate these data
        #       c. Store in database and add to current timestep cache
        #
        # Step 2:
        #   Thread 1:
        #       a. Get dispatch model inputs from data cache of current and/or previous timestep
        #       b. Call dispatch model using inputs
        #       c. Validate these data
        #       d. Store in database and add to current timestep cache
        #
        # Step 3:
        #   Thread 1:
        #       a. Set plant state and weather values in PySAM
        #       b. Call PySAM using inputs
        #       c. Validate these data
        #       d. Add data to cache and store in database
        #
        # Step 4:
        #   Thread 1:
        #       a. Update previous timestep cache with current timestep cache and then clear current cache


        # Step 3, Thread 1:
        # a. Set plant state and weather values in PySAM
        plant_state = self.pysam_wrap.GetSimulatedPlantState(self.validated_outputs_prev)      # for initializing next simulation from a prior one
        if plant_state is None:
            plant_state = self.pysam_wrap.GetDefaultPlantState()
        result = self.pysam_wrap.SetPlantState(plant_state)
        
        # b. Call PySAM using inputs
        datetime_now = datetime.datetime.now()
        datetime_start = RoundMinutes(datetime_now, 'down', self.timestep_simulation.seconds/60)    # the start of the time interval currently in
        # datetime_end = RoundMinutes(datetime_now, 'up', self.timestep_simulation.seconds/60)        # the end of the time interval currently in
        datetime_end = datetime_start + datetime.timedelta(minutes=60)        # just to see some values while testing at night
        tech_outputs = self.pysam_wrap.Simulate(datetime_start, datetime_end, self.timestep_simulation)
        print("Annual Energy [kWh]= ", tech_outputs["annual_energy"])

        # c. Validate these data
        # wanted_keys = ['time_hr', 'e_ch_tes', 'eta_therm', 'eta_field', 'P_out_net', 'tou_value', 'gen', 'q_dot_rec_inc', 'q_sf_inc', 'pricing_mult', 'beam']
        # wanted_keys = list(set(tech_outputs.keys()))       # all keys are wanted
        # wanted_outputs = dict((k, tech_outputs[k]) for k in wanted_keys if k in tech_outputs)
        wanted_outputs = tech_outputs
        wanted_outputs = {k:(list(v) if isinstance(v, tuple) else v) for (k,v) in wanted_outputs.items()}     # converts tuples to lists so they can be edited
        tic = time.process_time()
        validated_outputs = data_validator.validate(wanted_outputs, data_validator.pysam_schema)
        toc = time.process_time()
        print("Validation took {seconds:0.2f} seconds".format(seconds=toc-tic))

        # d. Add data to cache and store in database
        self.validated_outputs_prev = copy.deepcopy(validated_outputs)
        # pysam_table = models.PysamTable()
        # pysam_table.P_out_net = validated_outputs['P_out_net']          # NOT CORRECT, NEED TO ADD ACCORDING TO THE TIMESTEP INDEX
        # pysam_table.save()

        return 0

    def RunContinuously(self, update_interval=5):
        """Continuously get data from external plant and weather interfaces and run
        entire set of submodels, saving data to database
        
        Keyword arguments:
        update_interval -- [s] how frequently the interfaces and submodels are polled and run, respectively
        """
        looping_call = LoopingCall(self.RunOnce)
        time.sleep(update_interval - time.time() % update_interval)          # wait to start until it's an even clock interval
        looping_call.start(update_interval)
        reactor.run()

def MediateContinuously(update_interval=5):
    mediator = Mediator()
    mediator.RunContinuously(update_interval=update_interval)
    return False

# def MediateOnce():
#     """This will likely only be used for testing"""
#     mediator = Mediator()
#     mediator.RunOnce()
#     return False

def RoundMinutes(dt, direction, minute_resolution):
    new_minute = (dt.minute // minute_resolution + (1 if direction == 'up' else 0)) * minute_resolution
    new_time_old_seconds = dt + datetime.timedelta(minutes=new_minute - dt.minute)
    return new_time_old_seconds.replace(second=0, microsecond=0)

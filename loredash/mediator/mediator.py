import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import time
from datetime import datetime
from twisted.internet.task import LoopingCall
from twisted.internet import reactor
import PySAM_DAOTk.TcsmoltenSalt as pysam
from pathlib import Path
from mediator import data_validator, nested_inputs, models

class Mediator:
    def __init__(self):
        # initialize submodels and external interfaces
        pass
    
    def RunOnce(self):
        """For the current point in time, get data from external plant and weather interfaces and run
        entire set of submodels, saving data to database"""
        # print(datetime.now())
        # pass

        # The real code:
        # Step 1:
        #   Thread 1:
        #       a. If virtual plant, query database for needed inputs (or use cache from previous timestep)
        #       b. Call virtual/real plant to get plant operating state and any local weather data
        #       b. Validate these data
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
        #       a. Get PySAM inputs from data cache of current and/or previous timestep
        #       b. Call PySAM using inputs
        #       c. Validate these data
        #       d. Store in database and add to current timestep cache
        #
        # Step 4:
        #   Thread 1:
        #       a. Update previous timestep cache with current timestep cache and then clear current cache


        # Step 3, Thread 1:
        # b. Call PySAM using inputs
        parent_dir = str(Path(__file__).parents[1])
        weather_file = parent_dir+"/data/daggett_ca_34.865371_-116.783023_psmv3_60_tmy.csv"
        model_name = "MSPTSingleOwner"
        tech_model = pysam.default(model_name)
        tech_model.SolarResource.solar_resource_file = weather_file
        # NEED TO CHOOSE A SINGLE TIMESTEP INSTALL OF AN ANNUAL SIMULATION
        tech_model.execute(1) # 0 = verbosity level, or you can use 1 to show verbose output
        tech_outputs = tech_model.Outputs.export()

        # c. Validate these data
        wanted_keys = ['P_out_net']
        wanted_outputs = dict((k, tech_outputs[k]) for k in wanted_keys if k in tech_outputs)
        schema = nested_inputs.schemas['pysam_daotk']
        validated_outputs = data_validator.validate(wanted_outputs, schema)

        # d. Store in database and add to current timestep cache
        pysam_table = models.PysamTable()
        # pysam_table.P_out_net = validated_outputs['P_out_net']          # NOT CORRECT, NEED TO ADD ACCORDING TO THE TIMESTEP INDEX
        # pysam_table.save()

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

def MediateOnce():
    """This will likely only be used for testing"""
    mediator = Mediator()
    mediator.RunOnce()
    return False

def MediateContinuously(update_interval=5):
    mediator = Mediator()
    mediator.RunContinuously(update_interval=update_interval)
    return False

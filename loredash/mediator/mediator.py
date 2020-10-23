import time
from datetime import datetime
from twisted.internet.task import LoopingCall
from twisted.internet import reactor
# import multiprocessing

class Mediator:
    def __init__(self):
        # initialize submodels and external interfaces
        pass
    
    def RunOnce(self):
        """For the current point in time, get data from external plant and weather interfaces and run
        entire set of submodels, saving data to database"""
        print(datetime.now())
        pass

        # The real code:
        # Step 1:
        #   Thread 1:
        #       a. Call virtual/real plant to get plant operating state and any local weather data
        #       b. Validate these data
        #       c. Store in database
        #   Thread 2:
        #       a. Call module to retrieve weather data(s)
        #       b. Validate these data
        #       c. Store in database
        #
        # Step 2:
        #   Thread 1:
        #       a. Query database for dispatch model inputs
        #       b. Call dispatch model using inputs
        #       c. Validate these data
        #       d. Store in database
        #
        # Step 3:
        #   Thread 1:
        #       a. Query database for PySAM inputs
        #       b. Call PySAM using inputs
        #       c. Validate these data
        #       d. Store in database

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

import time
from datetime import datetime
from twisted.internet.task import LoopingCall
from twisted.internet import reactor
# import model and interface libraries

class Mediator:
    def __init__(self):
        # initialize submodels and external interfaces
        pass
    
    def MediateOnce(self):
        """For the current point in time, get data from external plant and weather interfaces and run
        entire set of submodels, saving data to database"""
        print(datetime.now())
        pass

    def MediateContinuously(self, update_interval=5):
        """Continuously get data from external plant and weather interfaces and run
        entire set of submodels, saving data to database
        
        Keyword arguments:
        update_interval -- [s] how frequently the interfaces and submodels are polled and run, respectively
        """

        looping_call = LoopingCall(self.MediateOnce)
        time.sleep(update_interval - time.time() % update_interval)          # wait to start until it's an even clock interval
        looping_call.start(update_interval)
        reactor.run()

if __name__ == '__main__':
    mediator = Mediator()
    mediator.MediateContinuously(update_interval=1)
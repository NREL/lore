from django.apps import AppConfig
import multiprocessing

class MediatorConfig(AppConfig):
    name = 'mediator'
    verbose_name = "Mediator"
    def ready(self):
        # startup code in here
        from mediator import mediator       # this needs to be down here, or you get the error "Apps aren't loaded yet"

        # THIS IS ALSO BEING CALLED BY THE BOKEH SERVER, AND IT SHOULDN'T BE
        # For testing, bypassing multiprocessing:
        mediator = mediator.Mediator()
        result = mediator.RunOnce()

        # THIS IS ALSO BEING CALLED BY THE BOKEH SERVER, SO AN ADDITIONAL THREAD IS CREATED DOING THE SAME THING:
        # update_interval = 10     # seconds
        # p = multiprocessing.Process(target=mediator.MediateContinuously, args=(update_interval,))
        # p.start()

        # This code works for running two simultaneous mediate processes parallel to main dashboard:
        # p = multiprocessing.Process(target=mediator.MediateContinuously, args=(1,))
        # p.start()

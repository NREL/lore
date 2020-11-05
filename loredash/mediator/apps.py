from django.apps import AppConfig

class MediatorConfig(AppConfig):
    name = 'mediator'
    verbose_name = "Mediator"
    def ready(self):
        # startup code in here
        from mediator import mediator       # this needs to be down here, or you get the error "Apps aren't loaded yet"
        mediator = mediator.Mediator()

        # 1. If the flux map file isn't already present, generate a new one
        # 

        # result = mediator.RunOnce()

        # update_interval = 5     # seconds
        # p = multiprocessing.Process(target=mediator.MediateContinuously, args=(update_interval,))
        # p.start()

        # This code works for running two simultaneous mediate processes parallel to main dashboard:
        # p = multiprocessing.Process(target=mediator.MediateContinuously, args=(1,))
        # p.start()

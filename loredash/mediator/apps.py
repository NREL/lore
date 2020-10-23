from django.apps import AppConfig
import mediator.views
# from mediator import mediator
# import multiprocessing

class MediatorConfig(AppConfig):
    name = 'mediator'

# update_interval = 10     # seconds
# p = multiprocessing.Process(target=mediator.MediateContinuously, args=(update_interval,))
# p.start()
# This code works for running two simultaneous mediate processes parallel to main dashboard:
# p = multiprocessing.Process(target=mediator.MediateContinuously, args=(1,))
# p.start()

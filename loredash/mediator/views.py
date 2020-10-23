from django.shortcuts import render
from mediator import mediator
import multiprocessing

# Create your views here.

update_interval = 5     # seconds
p = multiprocessing.Process(target=mediator.MediateContinuously, args=(update_interval,))
p.start()
# This code works for running two simultaneous mediate processes parallel to main dashboard:
# p = multiprocessing.Process(target=mediator.MediateContinuously, args=(1,))
# p.start()
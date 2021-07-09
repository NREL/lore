from django.core.management.base import BaseCommand
import os, sys
dir_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), # /commands
    '..',                                        # /management
    '..',                                        # /mediation
)
sys.path.insert(1, dir_path)
from mediation import mediator

class Command(BaseCommand):
    def handle(self, *args, **kwargs):
        mediator.run_lore()
        
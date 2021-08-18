from django.core.management.base import BaseCommand, CommandError

import datetime
from mediation import learning
import pytz
import sys

class Command(BaseCommand):
    help = 'Runs the learning model case study'

    def add_arguments(self, parser):
        parser.add_argument('input_noise', type=float)
        parser.add_argument('output_noise', type=float)
        parser.add_argument('replications', type=int)

    def handle(self, *args, **options):
        PARENT_DIR = "."
        model = learning.LearningModel(
            weather_file=PARENT_DIR + "/data/daggett_ca_34.865371_-116.783023_psmv3_60_tmy.csv",
            plant_design_path=PARENT_DIR + "/config/plant_design.json",
            mediator_params_path=PARENT_DIR + "/config/mediator_settings.json",
            dispatch_params_path=PARENT_DIR + "/config/dispatch_settings.json",
            datetime_start=datetime.datetime(2021, 6, 1, 0, 0, 0, tzinfo=pytz.UTC),
            datetime_end=datetime.datetime(2021, 6, 3, 0, 0, 0, tzinfo=pytz.UTC),
            outputs=['m_dot_rec', 'T_rec_out'],
        )
        model.run_case_study(
            input_noise=options["input_noise"],
            output_noise=options["output_noise"],
            replications=options["replications"],
        )
        return

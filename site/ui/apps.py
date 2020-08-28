from django.apps import AppConfig
from importlib import import_module
from django.conf import settings
from ui import mspt
import datetime

class UiConfig(AppConfig):
    name = 'ui'
    verbose_name = 'Dashboard App'

pysam_output = mspt.get_pysam_data()
pysam_output['time'] = list(pysam_output['time_hr'])
pysam_output['time'] = [datetime.timedelta(hours=int(x)) for x in pysam_output['time']]
pysam_output['time'] = list(map(lambda hr: hr + datetime.datetime(2010, 1, 1), pysam_output['time']))

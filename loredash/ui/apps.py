from django.apps import AppConfig
from importlib import import_module
from django.conf import settings

# global variables
pysam_output = None

class UiConfig(AppConfig):
    name = 'ui'
    verbose_name = 'Dashboard App'

from django.apps import AppConfig
from importlib import import_module
from django.conf import settings

# global variables for app


class UiConfig(AppConfig):
    name = 'ui'
    verbose_name = 'Dashboard App'

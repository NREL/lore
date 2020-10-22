# from ui import config, views

default_app_config = 'ui.apps.UiConfig'

# moved back to views.py to avoid a second call by the bokeh server
# config.pysam_output = views.getPysamData()
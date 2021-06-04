"""loredash URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from pathlib import Path
import datetime
import multiprocessing
from pyinstrument import Profiler       # can also profile Django, see: https://github.com/joerick/pyinstrument

from mediation import mediator
import mediation.plant as plant_

RUN_PROFILER = False

# TODO: Ensure database migration happens before this code is run. It is currently
# run on a 'migrate', which initially fails because no database has yet been created.
def init_and_mediate():
    parent_dir = str(Path(__file__).parents[1])
    default_weather_file = parent_dir + "/data/daggett_ca_34.865371_-116.783023_psmv3_60_tmy.csv"
    plant_config_path = parent_dir + "/data/plant_config.json"
    m = mediator.Mediator(
        params=mediator.mediator_params,
        plant_config_path = plant_config_path,
        plant_design=plant_.plant_design,                       # TODO: does this need to be a parameter?
        weather_file = default_weather_file,
        update_interval = datetime.timedelta(seconds = 5),
    )
    result = m.model_previous_day_and_add_to_db()
    # result = m.RunOnce()          #TODO: reenable and get this working
    return

if RUN_PROFILER:
    profiler = Profiler()
    profiler.start()

try:
    init_and_mediate()
except OSError as err:
    print("ERROR: OS error: {0}".format(err))
except Exception as err:
    print("ERROR: {0}".format(err))

if RUN_PROFILER:
    profiler.stop()
    profiler.open_in_browser()

# This is the main production code where the mediator runs continuously
# update_interval = 10     # seconds
# p = multiprocessing.Process(target=mediator.MediateContinuously, args=(update_interval,))
# p.start()

# This code adds another simultaneous mediate process (although likely not needed):
# p = multiprocessing.Process(target=mediator.MediateContinuously, args=(1,))
# p.start()
# ===/initialization=========================================================================

urlpatterns = [
    path('', include('ui.urls')),
    path('admin/', admin.site.urls),
]

if settings.DEBUG is True:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

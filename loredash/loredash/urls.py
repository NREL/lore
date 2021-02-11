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
from mediation import mediator
import multiprocessing
import datetime
# TODO(odow): the purpose of this code is to populate the database so we have
# things to plot. But it shouldn't go here, because this gets run on a `migrate`
# call, and initially, we don't have a database to store the results in! It
# should probably just go somewhere so it gets run when the root site is hit.
#
# python manage.py migrate
# python manage.py migrate
def _RunOnce():
    parent_dir = str(Path(__file__).parents[1])
    default_weather_file = parent_dir + "/data/daggett_ca_34.865371_-116.783023_psmv3_60_tmy.csv"
    plant_config_path = parent_dir + "/data/plant_config.json"
    m = mediator.Mediator(
        plant_config_path = plant_config_path,
        override_with_weather_file_location = False,
        weather_file = default_weather_file,
        preprocess_pysam = True,
        preprocess_pysam_on_init = True,
        update_interval = datetime.timedelta(seconds = 5),
        simulation_timestep = datetime.timedelta(minutes = 5),
    )
    result = m.ModelPreviousDayAndAddToDb()
    result = m.RunOnce()
    return
try:
    _RunOnce()
except Exception as err:
    print("Oops! Migration failed because we don't have a database yet. Try running that command again.")
    pass

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




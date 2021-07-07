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
from pyinstrument import Profiler       # can also profile Django, see: https://github.com/joerick/pyinstrument

from mediation import mediator

urlpatterns = [
    path('', include('ui.urls')),
    path('admin/', admin.site.urls),
]

if settings.DEBUG is True:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


# -- Initialization code, needed here in urls.py or it will be run twice, one of which by the bokeh server--
# NOTE: this may not be a problem anymore now that we're using settings.RUNNING_DEVSERVER,
#       so maybe could be moved out of urls.py? If so, verify that the Bokeh server isn't also running it.
RUN_PROFILER = False

if settings.RUNNING_DEVSERVER == True:
    if RUN_PROFILER:
        profiler = Profiler()
        profiler.start()

    try:
        mediator.init_and_mediate()
    except OSError as err:
        print("ERROR: OS error: {0}".format(err))
    except Exception as err:
        print("ERROR: {0}".format(err))

    if RUN_PROFILER:
        profiler.stop()
        profiler.open_in_browser()

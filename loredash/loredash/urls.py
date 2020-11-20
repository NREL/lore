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
from mediation import mediator
import multiprocessing

# ===Initialization code, put here so the Bokeh server django.setup() call doesn't execute it
# For testing, bypassing multiprocessing:
mediator = mediator.Mediator()
result = mediator.RunOnce()

# This is the main production code where the mediator runs continuously
# update_interval = 10     # seconds
# p = multiprocessing.Process(target=mediator.MediateContinuously, args=(update_interval,))
# p.start()

# This code adds another simultaneous mediate processes:
# p = multiprocessing.Process(target=mediator.MediateContinuously, args=(1,))
# p.start()
# ===/initialization=========================================================================

urlpatterns = [
    path('', include('ui.urls')),
    path('admin/', admin.site.urls),
]

if settings.DEBUG is True:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
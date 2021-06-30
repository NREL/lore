from django.contrib import admin

# Register your models here.
from .models import TechData, SolarForecastData

admin.site.register(TechData)
admin.site.register(SolarForecastData)

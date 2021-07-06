from django.contrib import admin

# Register your models here.
from .models import TechData, PlantStateData, WeatherData, SolarForecastData

admin.site.register(TechData)
admin.site.register(PlantStateData)
admin.site.register(WeatherData)
admin.site.register(SolarForecastData)

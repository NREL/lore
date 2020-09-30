from django.urls import path

from .views import dashboard_view, forecast_view, history_view, maintenance_view, settings_view

urlpatterns = [
    path(r'', dashboard_view, name='dashboard'),
    path(r'forecast', forecast_view, name='forecast'),
    path(r'history', history_view, name='history'),
    path(r'maintenance', maintenance_view, name='maintenance'),
    path(r'settings', settings_view, name='settings'),
]
from django.db import models
from django.conf import settings

class TechData(models.Model):                                                                               # SSC variable name [pre unit conversion]:
    timestamp = models.DateTimeField(verbose_name="Timestep end", primary_key=True)                         #   time_hr (end of timestep) [hr]
    E_tes_charged = models.FloatField(verbose_name="TES charge state [kWht]", default=None)                 #   e_ch_tes [MWht]
    eta_tower_thermal = models.FloatField(verbose_name="Tower thermal efficiency [-]", default=None)        #   eta_therm [-]
    eta_field_optical = models.FloatField(verbose_name="Field optical efficiency [-]", default=None)        #   eta_field [-]
    W_grid_no_derate = models.FloatField(verbose_name="Power to grid without derate [kWe]", default=None)   #   P_out_net [MWe]
    tou = models.FloatField(verbose_name="TOU value [-]", default=None)                                     #   tou_value [-]
    W_grid_with_derate = models.FloatField(verbose_name="Power to grid with derate [kWe]", default=None)    #   gen [kWe]
    Q_tower_incident = models.FloatField(verbose_name="Tower incident thermal power [kWt]", default=None)   #   q_dot_rec_inc [MWt]
    Q_field_incident = models.FloatField(verbose_name="Field incident thermal power [kWt]", default=None)   #   q_sf_inc [MWt]
    pricing_multiple = models.FloatField(verbose_name="Pricing multiple [-]", default=None)                 #   pricing_mult [-]
    dni = models.FloatField(verbose_name="DNI [W/m2]", default=None)                                        #   beam [W/m2]

    def __str__(self):
        """shown when entry is generically queried"""
        return str(self.timestamp)

class SolarForecastData(models.Model):
    forecast_made = models.DateTimeField(verbose_name="Forecast Made", db_index=True)
    forecast_for = models.DateTimeField(verbose_name="Forecast For", db_index=True)
    clear_sky = models.FloatField(verbose_name="Clear Sky [W/m2]", default=None)
    ratio = models.FloatField(verbose_name="Ratio [Predicted/Clear Sky]", default=None)

    def __str__(self):
        return str(self.forecast_for)

from django.db import models
from django.db import transaction
import datetime


#--------------PySAM Data----------------------------------
class PysamData(models.Model):                                                                              # SSC variables, pre unit conversion:
    timestamp = models.DateTimeField(verbose_name="Timestep end", db_index=True)                            #   time_hr [hr]
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

    # shown when entry is generically queried
    def __str__(self):
        return str(self.timestamp)



# THE BELOW TABLES ARE STILL USED BUT WILL BE REPLACED:

#--------------Dashboard Data------------------------------
class DashboardDataRTO(models.Model):
    timestamp = models.DateTimeField(verbose_name="Timestamp", db_index=True)
    actual = models.FloatField(verbose_name="Actual [MWe]", default=None)
    optimal = models.FloatField(verbose_name="Optimal [MWe]",default=None)
    scheduled = models.FloatField(verbose_name="Scheduled [MWe]", default=None)
    field_operation_generated = models.FloatField(verbose_name="Field Operation Generated [MWt]", default=None)
    field_operation_available = models.FloatField(verbose_name="Field Operation Available [MWt]", default=None)

    # define what is shown when entry is generically queried
    def __str__(self):
        return str(self.timestamp)


#----------Forecasts Market Data----------------------------
class ForecastsMarketData(models.Model):
    timestamp = models.DateTimeField(verbose_name="Timestamp", db_index=True)
    market_forecast = models.FloatField(verbose_name="Market Forecast [-]", default=None)
    ci_plus = models.FloatField(verbose_name="CI+ [%]", default=None)
    ci_minus = models.FloatField(verbose_name="CI- [%]", default=None)

    def __str__(self):
        return str(self.timestamp)

#-----------Forecasts Solar Data----------------------------
class ForecastsSolarData(models.Model):
    timestamp = models.DateTimeField(verbose_name="Timestamp", db_index=True)
    clear_sky = models.FloatField(verbose_name="Clear Sky [W/m2]", default=None)
    nam = models.FloatField(verbose_name="NAM [W/m2]", default=None)
    nam_plus = models.FloatField(verbose_name="NAM+ [%]", default=None)
    nam_minus = models.FloatField(verbose_name="NAM- [%]", default=None)
    rap = models.FloatField(verbose_name="RAP [W/m2]", default=None)
    rap_plus = models.FloatField(verbose_name="RAP+ [%]", default=None)
    rap_minus = models.FloatField(verbose_name="RAP- [%]", default=None)
    hrrr = models.FloatField(verbose_name="HRRR [W/m2]", default=None)
    hrrr_plus = models.FloatField(verbose_name="HRRR+ [%]", default=None)
    hrrr_minus = models.FloatField(verbose_name="HRRR- [%]", default=None)
    gfs = models.FloatField(verbose_name="GFS [W/m2]", default=None)
    gfs_plus = models.FloatField(verbose_name="GFS+ [%]", default=None)
    gfs_minus = models.FloatField(verbose_name="GFS- [%]", default=None)
    ndfd = models.FloatField(verbose_name="NDFD [W/m2]", default=None)
    ndfd_plus = models.FloatField(verbose_name="NDFD+ [%]", default=None)
    ndfd_minus = models.FloatField(verbose_name="NDFD- [%]", default=None)

    def __str__(self):
        return str(self.timestamp)

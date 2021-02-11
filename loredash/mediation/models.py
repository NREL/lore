from django.db import models
from django.conf import settings

#--------------PySAM Data----------------------------------
class PysamData(models.Model):                                                                              # SSC variables, pre unit conversion:
    timestamp = models.DateTimeField(verbose_name="Timestep end", primary_key=True)                         #   time_hr (end of timestep) [hr]
                                                                                                            #    (db_index=True is likely redundant to primary_key=True)
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


# I kind of gave up here with trying to enforce only one entry in this model/table. However, this
#  should work unless someone explicity specifies a site_id, and one other than '1'.
#  See these links for more robust ideas:
#   https://stackoverflow.com/a/2106836                         (not sure what this means)
#   https://stackoverflow.com/a/4888467                         (I can't get this idea to work)
#   https://docs.djangoproject.com/en/3.1/ref/contrib/sites/    (more background on the 'sites' framework)
#   https://djangopackages.org/grids/g/live-setting/            (Django-Constance looks good, but maybe overkill?)
class PlantConfig(models.Model):
    site_id = models.IntegerField(default=settings.SITE_ID, primary_key=True)
    name = models.CharField(max_length=255, verbose_name="Plant name", default='plant_name')                        # max_length of 255 is a safe constraint
    latitude = models.FloatField(verbose_name="Latitude, degrees North [deg]", default=-999)
    longitude = models.FloatField(verbose_name="Longitude, degrees East [deg]", default=-999)
    elevation = models.FloatField(verbose_name="Elevation above sea level [m]", default=-999)
    timezone = models.FloatField(verbose_name="Timezone, UTC offset [hr]", default=-999)

    # shown when entry is generically queried
    def __str__(self):
        return str(self.name)

#-----------Forecasts Solar Data----------------------------
class SolarForecasts(models.Model):
    timestamp = models.DateTimeField(verbose_name="Timestamp", db_index=True)
    clear_sky = models.FloatField(verbose_name="Clear Sky [W/m2]", default=None)
    ndfd = models.FloatField(verbose_name="NDFD [W/m2]", default=None)

    def __str__(self):
        return str(self.timestamp)

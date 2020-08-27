from django.db import models
from django.db import transaction
import datetime

# Load PySam Data at server load instead of session load
from ui import mspt
pysam_output = mspt.get_pysam_data()
pysam_output['time'] = list(pysam_output['time_hr'])
pysam_output['time'] = [datetime.timedelta(hours=int(x)) for x in pysam_output['time']]
pysam_output['time'] = list(map(lambda hr: hr + datetime.datetime(2010, 1, 1), pysam_output['time']))


# New plot tables:
#--------------Dashboard Data------------------------------
class DashboardDataRTO(models.Model):
    timestamp = models.DateTimeField(verbose_name="Timestamp")
    actual = models.FloatField(verbose_name="Actual [MWe]", default=None)
    optimal = models.FloatField(verbose_name="Optimal [MWe]",default=None)
    scheduled = models.FloatField(verbose_name="Scheduled [MWe]", default=None)
    field_operation_generated = models.FloatField(verbose_name="Field Operation Generated [MWt]", default=None)
    field_operation_available = models.FloatField(verbose_name="Field Operation Available [MWt]", default=None)

    def __str__(self):
        return str(self.timestamp)


#----------Forcasts Market Data----------------------------
class ForecastsMarketData(models.Model):
    timestamp = models.DateTimeField(verbose_name="Timestamp")
    market_forecast = models.FloatField(verbose_name="Market Forcast [-]", default=None)
    ci_plus = models.FloatField(verbose_name="CI+ [%]", default=None)
    ci_minus = models.FloatField(verbose_name="CI- [%]", default=None)

    def __str__(self):
        return str(self.timestamp)

#-----------Forcasts Solar Data----------------------------
class ForecastsSolarData(models.Model):
    timestamp = models.DateTimeField(verbose_name="Timestamp")
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

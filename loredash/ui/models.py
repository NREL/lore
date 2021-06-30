# from django.db import models

# class DashboardDataRTO(models.Model):
#     timestamp = models.DateTimeField(verbose_name="Timestamp", db_index=True)
#     actual = models.FloatField(verbose_name="Actual [MWe]", default=None)
#     optimal = models.FloatField(verbose_name="Optimal [MWe]",default=None)
#     scheduled = models.FloatField(verbose_name="Scheduled [MWe]", default=None)
#     field_operation_generated = models.FloatField(verbose_name="Field Operation Generated [MWt]", default=None)
#     field_operation_available = models.FloatField(verbose_name="Field Operation Available [MWt]", default=None)

#     # define what is shown when entry is generically queried
#     def __str__(self):
#         return str(self.timestamp)

# class ForecastsMarketData(models.Model):
#     timestamp = models.DateTimeField(verbose_name="Timestamp", db_index=True)
#     market_forecast = models.FloatField(verbose_name="Market Forecast [-]", default=None)
#     ci_plus = models.FloatField(verbose_name="CI+ [%]", default=None)
#     ci_minus = models.FloatField(verbose_name="CI- [%]", default=None)

#     def __str__(self):
#         return str(self.timestamp)

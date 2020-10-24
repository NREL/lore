from django.db import models

#--------------PySAM Data------------------------------
class PysamTable(models.Model):
    timestamp = models.DateTimeField(verbose_name="Timestamp", db_index=True)
    P_out_net = models.FloatField(verbose_name="P_out_net [MWe]", default=None)
    
    # define what is shown when entry is generically queried
    def __str__(self):
        return str(self.timestamp)

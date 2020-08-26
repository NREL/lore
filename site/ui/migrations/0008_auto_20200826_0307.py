# Generated by Django 3.0.3 on 2020-08-26 03:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ui', '0007_auto_20200824_1803'),
    ]

    operations = [
        migrations.AlterField(
            model_name='dashboarddatarto',
            name='actual',
            field=models.FloatField(default=None, verbose_name='Actual [MWe]'),
        ),
        migrations.AlterField(
            model_name='dashboarddatarto',
            name='field_operation_available',
            field=models.FloatField(default=None, verbose_name='Field Operation Available [MWt]'),
        ),
        migrations.AlterField(
            model_name='dashboarddatarto',
            name='field_operation_generated',
            field=models.FloatField(default=None, verbose_name='Field Operation Generated [MWt]'),
        ),
        migrations.AlterField(
            model_name='dashboarddatarto',
            name='optimal',
            field=models.FloatField(default=None, verbose_name='Optimal [MWe]'),
        ),
        migrations.AlterField(
            model_name='dashboarddatarto',
            name='scheduled',
            field=models.FloatField(default=None, verbose_name='Scheduled [MWe]'),
        ),
        migrations.AlterField(
            model_name='forecastsmarketdata',
            name='ci_minus',
            field=models.FloatField(default=None, verbose_name='CI- [%]'),
        ),
        migrations.AlterField(
            model_name='forecastsmarketdata',
            name='ci_plus',
            field=models.FloatField(default=None, verbose_name='CI+ [%]'),
        ),
        migrations.AlterField(
            model_name='forecastsmarketdata',
            name='market_forecast',
            field=models.FloatField(default=None, verbose_name='Market Forcast [-]'),
        ),
        migrations.AlterField(
            model_name='forecastssolardata',
            name='clear_sky',
            field=models.FloatField(default=None, verbose_name='Clear Sky [W/m2]'),
        ),
        migrations.AlterField(
            model_name='forecastssolardata',
            name='gfs',
            field=models.FloatField(default=None, verbose_name='GFS [W/m2]'),
        ),
        migrations.AlterField(
            model_name='forecastssolardata',
            name='gfs_minus',
            field=models.FloatField(default=None, verbose_name='GFS- [%]'),
        ),
        migrations.AlterField(
            model_name='forecastssolardata',
            name='gfs_plus',
            field=models.FloatField(default=None, verbose_name='GFS+ [%]'),
        ),
        migrations.AlterField(
            model_name='forecastssolardata',
            name='hrrr',
            field=models.FloatField(default=None, verbose_name='HRRR [W/m2]'),
        ),
        migrations.AlterField(
            model_name='forecastssolardata',
            name='hrrr_minus',
            field=models.FloatField(default=None, verbose_name='HRRR- [%]'),
        ),
        migrations.AlterField(
            model_name='forecastssolardata',
            name='hrrr_plus',
            field=models.FloatField(default=None, verbose_name='HRRR+ [%]'),
        ),
        migrations.AlterField(
            model_name='forecastssolardata',
            name='nam',
            field=models.FloatField(default=None, verbose_name='NAM [W/m2]'),
        ),
        migrations.AlterField(
            model_name='forecastssolardata',
            name='nam_minus',
            field=models.FloatField(default=None, verbose_name='NAM- [%]'),
        ),
        migrations.AlterField(
            model_name='forecastssolardata',
            name='nam_plus',
            field=models.FloatField(default=None, verbose_name='NAM+ [%]'),
        ),
        migrations.AlterField(
            model_name='forecastssolardata',
            name='ndfd',
            field=models.FloatField(default=None, verbose_name='NDFD [W/m2]'),
        ),
        migrations.AlterField(
            model_name='forecastssolardata',
            name='ndfd_minus',
            field=models.FloatField(default=None, verbose_name='NDFD- [%]'),
        ),
        migrations.AlterField(
            model_name='forecastssolardata',
            name='ndfd_plus',
            field=models.FloatField(default=None, verbose_name='NDFD+ [%]'),
        ),
        migrations.AlterField(
            model_name='forecastssolardata',
            name='rap',
            field=models.FloatField(default=None, verbose_name='RAP [W/m2]'),
        ),
        migrations.AlterField(
            model_name='forecastssolardata',
            name='rap_minus',
            field=models.FloatField(default=None, verbose_name='RAP- [%]'),
        ),
        migrations.AlterField(
            model_name='forecastssolardata',
            name='rap_plus',
            field=models.FloatField(default=None, verbose_name='RAP+ [%]'),
        ),
    ]

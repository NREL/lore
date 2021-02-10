# Generated by Django 2.2.2 on 2020-11-24 18:03

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mediation', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='PysamData',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('timestamp', models.DateTimeField(db_index=True, verbose_name='Timestep end')),
                ('E_tes_charged', models.FloatField(default=None, verbose_name='TES charge state [kWht]')),
                ('eta_tower_thermal', models.FloatField(default=None, verbose_name='Tower thermal efficiency [-]')),
                ('eta_field_optical', models.FloatField(default=None, verbose_name='Field optical efficiency [-]')),
                ('W_grid_no_derate', models.FloatField(default=None, verbose_name='Power to grid without derate [kWe]')),
                ('tou', models.FloatField(default=None, verbose_name='TOU value [-]')),
                ('W_grid_with_derate', models.FloatField(default=None, verbose_name='Power to grid with derate [kWe]')),
                ('Q_tower_incident', models.FloatField(default=None, verbose_name='Tower incident thermal power [kWt]')),
                ('Q_field_incident', models.FloatField(default=None, verbose_name='Field incident thermal power [kWt]')),
                ('pricing_multiple', models.FloatField(default=None, verbose_name='Pricing multiple [-]')),
                ('dni', models.FloatField(default=None, verbose_name='DNI [W/m2]')),
            ],
        ),
        migrations.DeleteModel(
            name='PysamTable',
        ),
    ]
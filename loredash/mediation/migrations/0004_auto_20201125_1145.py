# Generated by Django 2.2.2 on 2020-11-25 11:45

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mediation', '0003_auto_20201124_1807'),
    ]

    operations = [
        migrations.AlterField(
            model_name='pysamdata',
            name='timestamp',
            field=models.DateTimeField(primary_key=True, serialize=False, verbose_name='Timestep end'),
        ),
    ]
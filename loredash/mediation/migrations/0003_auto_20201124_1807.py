# Generated by Django 2.2.2 on 2020-11-24 18:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mediation', '0002_auto_20201124_1803'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='pysamdata',
            name='id',
        ),
        migrations.AlterField(
            model_name='pysamdata',
            name='timestamp',
            field=models.DateTimeField(db_index=True, primary_key=True, serialize=False, verbose_name='Timestep end'),
        ),
    ]

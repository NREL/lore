# Generated by Django 2.2.2 on 2021-02-03 14:29

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('mediation', '0005_plantconfig'),
    ]

    operations = [
        migrations.AlterField(
            model_name='plantconfig',
            name='site',
            field=models.OneToOneField(on_delete=django.db.models.deletion.PROTECT, to='sites.Site'),
        ),
    ]
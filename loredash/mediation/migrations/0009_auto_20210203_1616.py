# Generated by Django 2.2.2 on 2021-02-03 16:16

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mediation', '0008_auto_20210203_1518'),
    ]

    operations = [
        migrations.AlterModelManagers(
            name='plantconfig',
            managers=[
            ],
        ),
        migrations.RemoveField(
            model_name='plantconfig',
            name='id',
        ),
        migrations.RemoveField(
            model_name='plantconfig',
            name='site',
        ),
        migrations.AddField(
            model_name='plantconfig',
            name='site_id',
            field=models.IntegerField(default=1, primary_key=True, serialize=False),
        ),
    ]
# Generated by Django 2.1.3 on 2018-12-19 21:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('part1', '0002_auto_20181220_0025'),
    ]

    operations = [
        migrations.AlterField(
            model_name='metasurvey',
            name='resized_survey',
            field=models.ImageField(blank=True, null=True, upload_to='resized_survey/'),
        ),
    ]
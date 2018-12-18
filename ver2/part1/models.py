from django.db import models
from .fields import ThumbnailImageField
from django.utils.timezone import now

# Create your models here.
class MetaSurvey(models.Model):
    title = models.CharField(max_length=50)
    survey = ThumbnailImageField(upload_to='survey/%Y/%m')
    data = models.FileField(upload_to="data/%d", null=True, blank=True)

    resized_survey = models.ImageField(upload_to='resized_survey/%d', null=True, blank=True)
    upload_date = models.DateTimeField('Upload Date', auto_now_add=True)

    class Meta:
        ordering = ['title']

    def __str__(self):
        return self.title

    # resized_survey = models.ImageField(upload_to='resized_survey/%d', null=True, blank=True)
    # resized_survey = models.ImageField(upload_to="resized_survey/%d", null=True, blank=True)
    # resized_survey = ThumbnailImageField(upload_to='resized_survey/%d')

class Unziped_data(models.Model):
    meta_survey = models.ForeignKey(MetaSurvey, on_delete=models.CASCADE)
    image = models.ImageField(upload_to="unzip/%d", null=True, blank=True)


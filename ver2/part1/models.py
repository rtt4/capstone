from django.db import models
from .fields import ThumbnailImageField
from django.utils.timezone import now

# Create your models here.
class MetaSurvey(models.Model):
    survey = models.ImageField(upload_to="survey/%d", null=True, blank=True)
    data = models.FileField(upload_to="data/%d", null=True, blank=True)
    upload_date = models.DateTimeField('Upload Date', auto_now_add=True)
    resized_survey = models.ImageField(upload_to='resized_survey/%d', null=True, blank=True)
    # resized_survey = models.ImageField(upload_to="resized_survey/%d", null=True, blank=True)
    # resized_survey = ThumbnailImageField(upload_to='resized_survey/%d')

class Unziped_data(models.Model):
    image = models.ImageField(upload_to="unzip/%d", null=True, blank=True)

# class DataSurvey(models.Model):
#     data = models.FileField(upload_to="unzip_data/%d", null=True, blank=True)
#
# class Question(models.Model):
#     survey = models.ForeignKey(
#         'MetaSurvey',
#         on_delete=models.CASCADE,
#     )
#     question_text = models.CharField(max_length=200)
#     x = models.IntegerField(default=0)
#     y = models.IntegerField(default=0)
#     h = models.IntegerField(default=0)
#     w = models.IntegerField(default=0)
#
#     def __str__(self):
#         return self.question_text
#
#
# class Answer(models.Model):
#     # https://docs.djangoproject.com/en/1.11/ref/models/fields/#django.db.models.ForeignKey
#     question = models.ForeignKey(
#         'Question',
#         on_delete=models.CASCADE,
#     )
#
#     answer_text = models.TextField
#     votes = models.IntegerField(default=0)
#     x = models.IntegerField(default=0)
#     y = models.IntegerField(default=0)
#     h = models.IntegerField(default=0)
#     w = models.IntegerField(default=0)
#
#     # image = ThumbnailImageField(upload_to='photo/%Y/%m')
#     upload_date = models.DateTimeField('Upload Date', auto_now_add=True)
#
#     def __str__(self):
#         return self.answer_text
#
#     # def get_absolute_url(self):
#     #     return reverse('photo:photo_detail', args=(self.id,))


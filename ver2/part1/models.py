from django.db import models
from django.utils import timezone
from django.urls import reverse

# Create your models here.
class MetaSurvey(models.Model):
    # survey = models.FileField(upload_to='survey/%Y/%m/%d')
    survey = models.ImageField(upload_to="survey/%d", null=True, blank=True)
    data = models.FileField(upload_to="data/%d", null=True, blank=True)
    # class Meta:
    #     db_table = 'meta_table'


class Question(models.Model):
    survey = models.ForeignKey(
        'MetaSurvey',
        on_delete=models.CASCADE,
    )
    question_text = models.CharField(max_length=200)
    x = models.IntegerField(default=0)
    y = models.IntegerField(default=0)
    h = models.IntegerField(default=0)
    w = models.IntegerField(default=0)

    def __str__(self):
        return self.question_text

    # def get_absolute_url(self):
    #     return reverse('photo:album_detail', args=(self.id,))   #수정 필요


class Answer(models.Model):
    # https://docs.djangoproject.com/en/1.11/ref/models/fields/#django.db.models.ForeignKey
    question = models.ForeignKey(
        'Question',
        on_delete=models.CASCADE,
    )

    answer_text = models.TextField
    votes = models.IntegerField(default=0)
    x = models.IntegerField(default=0)
    y = models.IntegerField(default=0)
    h = models.IntegerField(default=0)
    w = models.IntegerField(default=0)

    # image = ThumbnailImageField(upload_to='photo/%Y/%m')
    upload_date = models.DateTimeField('Upload Date', auto_now_add=True)

    def __str__(self):
        return self.answer_text

    # def get_absolute_url(self):
    #     return reverse('photo:photo_detail', args=(self.id,))


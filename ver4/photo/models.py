from django.db import models

# Create your models here.
from django.urls import reverse
from .fields import ThumbnailImageField  #1

class Album(models.Model):
    name = models.CharField(max_length=50)
    description = models.CharField('One Line Description', max_length=100, blank=True)

    class Meta:
        ordering = ['name']

    def __str__(self):
        return self.name

    #2 이 메소드가 정의된 객체를 지칭하는 URL 반환
    def get_absolute_url(self):
        return reverse('photo:album_detail', args=(self.id,))   # 예를 들어 /photo/album/99/ 형식


class Photo(models.Model):
    album = models.ForeignKey(Album, on_delete=models.CASCADE)
    title = models.CharField(max_length=50)
    image = ThumbnailImageField(upload_to='photo/%Y/%m')
    description = models.TextField('Photo Description', blank=True)
    upload_date = models.DateTimeField('Upload Date', auto_now_add=True)

    #7
    class Meta:
        ordering = ['title']

    def __str__(self):
        return self.title

    #8
    def get_absolute_url(self):
        return reverse('photo:photo_detail', args=(self.id,))
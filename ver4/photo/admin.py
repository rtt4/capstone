from django.contrib import admin
from .models import Album, Photo

# Register your models here.

class PhotoInline(admin.StackedInline): # 세로로 나열, TabularInline은 테이블 모양처럼 행으로 나열
    model = Photo   # 추가로 보여지는 테이블
    extra = 2       # 이미 입력된 객체 외 추가로 입력할 수 있는 Photo 테이블 객체의 수는 2개


class AlbumAdmin(admin.ModelAdmin):
    inlines = [PhotoInline]     # 앨범 객체 보여줄 때 PhotoInline 클래스에서 정의한 사항을 같이 보여줌.
    list_display = ('name', 'description')


class PhotoAdmin(admin.ModelAdmin):
    list_display = ('title', 'upload_date')


admin.site.register(Album, AlbumAdmin)
admin.site.register(Photo, PhotoAdmin)

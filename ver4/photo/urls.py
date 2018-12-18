from django.urls import path
from django.conf.urls import url

from .views import *
from . import views

app_name = 'photo'
urlpatterns = [

    url(r'^p5/$', views.p5, name='p5'),

    # Example: /
    path('', AlbumLV.as_view(), name='index'),

    # Example: /album/, same as /
    path('album/', AlbumLV.as_view(), name='album_list'),

    # Example: /album/99/
    path('album/<int:pk>/', AlbumDV.as_view(), name='album_detail'),

    # Example: /photo/99/
    path('photo/<int:pk>/', PhotoDV.as_view(), name='photo_detail'),

    # Example: /album/add/
    path('album/add/', AlbumPhotoCV.as_view(), name="album_add",),

    # Example: /photo/add/
    path('photo/add/', PhotoCreateView.as_view(), name="photo_add",),


    # 1차적으로 필요없는 부분
    # Example: /album/change/
    path('album/change/', AlbumChangeLV.as_view(), name="album_change",),

    # Example: /album/99/update/
    path('album/<int:pk>/update/', AlbumPhotoUV.as_view(), name="album_update",),

    # Example: /album/99/delete/
    path('album/<int:pk>/delete/', AlbumDeleteView.as_view(), name="album_delete",),


    # Example: /photo/change/
    path('photo/change/', PhotoChangeLV.as_view(), name="photo_change",),

    # Example: /photo/99/update/
    path('photo/<int:pk>/update/', PhotoUpdateView.as_view(), name="photo_update",),

    # Example: /photo/99/delete/
    path('photo/<int:pk>/delete/', PhotoDeleteView.as_view(), name="photo_delete",),
]


from django.conf.urls import url
from django.urls import path

from . import views

urlpatterns = [
    url(r'^$', views.p0, name='p0'),
    url(r'^p1/$', views.p1, name='p1'),
    url(r'^p5/$', views.p5, name='p5'),
    path('download/', views.download, name='download')
]
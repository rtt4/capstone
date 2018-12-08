from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.p0, name='p0'),
    url(r'^p1/$', views.p1, name='p1'),
    # url(r'^p2/(?P<pk>\d+)/$', views.p2, name='p2'),
    # url(r'^p2/$', views.p2, name='p2'),
    url(r'^p4/$', views.p4, name='p4'),
    # url(r'^p5/(?P<pk>\d+)/$', views.p5, name='p5'),
    url(r'^p5/$', views.p5, name='p5'),
]
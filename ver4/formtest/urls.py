from django.conf.urls import url
from . import views

urlpatterns = [
    url('^$', views.get_name, name='get_name'),
    url('^/your_name/$', views.your_name, name='your_name'),
]
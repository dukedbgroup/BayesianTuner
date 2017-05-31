from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^echo/$', views.echo, name='echo'),
    url(r'^connect_db/$', views.connect_db, name='connect_db')
]

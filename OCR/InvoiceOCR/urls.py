from django.urls import path
from . import views

urlpatterns = [
    path('iocr/', views.index, name='index'),
]

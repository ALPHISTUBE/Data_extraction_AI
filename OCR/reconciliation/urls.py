from django.urls import path
from . import views

urlpatterns = [
    path('br/', views.reconcile_files, name='reconcile_files'),
]

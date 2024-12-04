from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('process-data/', views.predict_sleep_quality, name="process-data"),
]


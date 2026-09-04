from django.urls import path
from . import views  # Import views from the same directory

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict_word, name='predict_word'),
    path('translate/', views.translate_text, name='translate_text'),
]

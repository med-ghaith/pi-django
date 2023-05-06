from django.urls import path
from . import views

urlpatterns = [
    path('login/', views.loginPage, name='login'),
    path('logout/', views.logoutUser, name='logout'),
    path('register/', views.registerPage, name='register'),
    path('', views.home, name="home"),
    path('predict/', views.predict_view, name="predict"),
    path('recommend/', views.recommendation_view, name='recommendation_view'),
    
]
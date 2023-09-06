from django.contrib import admin
from django.urls import path
from . import controller

urlpatterns = [
    path('signup/',controller.signup),
    path('login/',controller.login),
    path('logout/',controller.logout),
]

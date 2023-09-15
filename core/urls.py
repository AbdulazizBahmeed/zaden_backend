from django.contrib import admin
from django.urls import path
from . import controller

urlpatterns = [
    path('signup/',controller.signup),
    path('login/',controller.login),
    path('google-login/',controller.google_login),
    path('logout/',controller.logout),
    path('verify-login/',controller.verify_login),
]

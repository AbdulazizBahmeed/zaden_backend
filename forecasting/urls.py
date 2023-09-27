from django.contrib import admin
from django.urls import path
from . import controller

urlpatterns = [
    path('upload-file/',controller.upload),
    path('list-files/',controller.list_files),
]

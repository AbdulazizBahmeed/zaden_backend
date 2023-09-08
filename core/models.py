from django.db import models
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    username = None
    first_name = None
    last_name = None
    fullname = models.CharField(max_length=255, null=True,blank=True)
    email = models.EmailField("email", unique=True)
    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = []

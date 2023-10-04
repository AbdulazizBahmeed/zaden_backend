from io import BytesIO
from django.db import models
from django.contrib.auth import get_user_model
from django.utils import timezone
import requests


class File(models.Model):
    file_name = models.CharField(max_length=255)
    uuid = models.UUIDField()
    owner = models.ForeignKey(get_user_model(), related_name="files",on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    def as_dict(self):
        return {
            "id": self.id,
            "name": self.file_name,
            "date" : self.created_at.astimezone(timezone.get_current_timezone()).strftime("%Y-%m-%d %I:%M%p"),
            # "date": localtime(self.created_at).strftime("%Y-%m-%d %I:%M %p")
        }
    
    def file(self):
        url = f'https://kpnzs85sk8.execute-api.ap-northeast-2.amazonaws.com/download-api/zaden-bucket/{self.uuid}.xlsx'
        response = requests.get(url)
        file = BytesIO(response.content)
        return file

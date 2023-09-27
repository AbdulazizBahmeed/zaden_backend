from django.db import models
from django.contrib.auth import get_user_model

class File(models.Model):
    file = models.FileField(upload_to="")
    owner = models.ForeignKey(get_user_model(), related_name="files",on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    def as_dict(self):
        return {
            "id": self.id,
            "name": self.file.name,
            "date": self.created_at.strftime("%Y-%m-%d %H:%M:%S")
        }

from django.contrib.auth.backends import ModelBackend
from django.contrib.auth import get_user_model
from django.http import JsonResponse

class EmailBackend(ModelBackend):
    def authenticate(self, request, email=None, password=None, **kwargs): #slef denotes for having an instance 
        UserModel = get_user_model()
        try:
            user = UserModel.objects.get(email=email)
        except UserModel.DoesNotExist:
            return None
        else:
            if user.check_password(password):
                return user
            else:
                return None
            
    def get_user(self, user_id):
        UserModel = get_user_model()
        try:
            return UserModel.objects.get(pk=user_id)
        except UserModel.DoesNotExist:
            return None

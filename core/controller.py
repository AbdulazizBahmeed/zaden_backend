from django.contrib.auth import authenticate, login as login_user, logout as logout_user
from django.http import JsonResponse
from django.contrib.auth import get_user_model
import re


def signup(req):
    if req.method=="POST":
        email =email=req.POST.get("email")
        password =req.POST.get("password")
        if validaite_email(email) and validaite_passowrd(password):
            UserModel = get_user_model()
            user = UserModel(email=email)
            user.set_password(password)
            try:
                user.save()
            except:
                return JsonResponse({
                    "status":False,
                    "message":"User credentials already used"
                })
            login_user(req,user)
            return JsonResponse({
                "status":True,
                "message":"user signed up succefully",
                })
        else:
            return JsonResponse(
                {
                    "status":False,
                    "message":"wrong email or password foramt",
                },status =400
            )
    else:
        return JsonResponse({
            "status":False,
            "message":"wrong method"
        },status=405)

def login(req):
    if req.method=="POST":
        email =email=req.POST.get("email")
        password =req.POST.get("password")
        user = authenticate(email=email,password=password)
        if user is not None:
            login_user(req,user)
            return JsonResponse({
                "status":True,
                "message":"User Logged In Succefully"
            }
            )
        else:
            return JsonResponse({
            "status":False,
            "message":"there is no user with that credentials"
        },status=400)
    else:
        return JsonResponse({
            "status":False,
            "message":"wrong method"
        },status=405)

def logout(req):
    if req.method=="GET":
        logout_user(req)
        return JsonResponse({
            "message":"logged out succefully"
        })
    else:
        return JsonResponse({
            "status":False,
            "message":"wrong method"
        },status=405)

def validaite_email(email):
    pattern = "[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,3}"
    if email is not None and re.match(email,pattern):
        return True
    else:
        return False

def validaite_passowrd(password):
    pattern = "[a-zA-Z0-9]{6,}"
    if password is not None and re.match(password,pattern):
        return True
    else:
        return False
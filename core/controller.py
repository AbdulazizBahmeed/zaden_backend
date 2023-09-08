from django.contrib.auth import authenticate, login as login_user, logout as logout_user
from django.http import JsonResponse
from django.contrib.auth import get_user_model
import re
from django.db.utils import IntegrityError

def validaite_email(email):
    pattern = "^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,3}$"
    if email is not None and re.match(pattern,email):
        return True
    else:
        return False

def validaite_passowrd(password):
    pattern = "^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)[A-Za-z\d]{6,}$"
    if password is not None and re.match(pattern,password):
        return True
    else:
        return False

def signup(req):
    if req.method=="POST":
        email =email=req.POST.get("email")
        password =req.POST.get("password")
        fullname =req.POST.get("fullname")

        if validaite_email(email) and validaite_passowrd(password):
            UserModel = get_user_model()
            user = UserModel(email=email,fullname=fullname)
            user.set_password(password)
            try:
                user.save()
            except IntegrityError:
                return JsonResponse({
                    "status":False,
                    "message":"User credentials already used"
                },status=400)
            login_user(req,user)
            return JsonResponse({
                "status":True,
                "message":"user signed up succefully",
                })
        else:
            return JsonResponse(
                {
                    "status":False,
                    "message":"wrong email or password format",
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
                "message":"User Logged In Succefully",
                "fullname": user.fullname,
                "email": user.email,
                "user_id": user.id
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
    
from django.contrib.auth import authenticate, login as login_user, logout as logout_user
from django.http import JsonResponse
from django.contrib.auth import get_user_model
import re
from django.db.utils import IntegrityError
import json

def validaite_email(email):
    pattern = "^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,3}$"
    if email is not None and re.match(pattern, email):
        return True
    else:
        return False


def validaite_passowrd(password):
    pattern = "^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)[A-Za-z\d]{6,}$"
    if password is not None and re.match(pattern, password):
        return True
    else:
        return False


def signup(req):
    if req.method=="POST":
        json_data = json.loads(req.body)
        email = json_data.get("email")
        password =json_data.get("password")
        fullname = json_data.get("fullname")

        if validaite_email(email) and validaite_passowrd(password):
            user_model = get_user_model()
            user = user_model(email=email, fullname=fullname)
            user.set_password(password)
            try:
                user.save()
            except IntegrityError:
                return JsonResponse({
                    "status": False,
                    "message": "يوجد حساب مسجل بهذا البريد الإلكتروني"
                }, status=400)
            login_user(req, user)
            return JsonResponse({
                "status": True,
                "message": "تم التسجيل بنجاح",
                "fullname": user.fullname,
                "email": user.email,
                "picture": user.picture,
                "user_id": user.id
            })
        else:
            return JsonResponse(
                {
                    "status": False,
                    "message": "خطأ في صيغة الايميل او كلمة السر",
                }, status=400
            )
    else:
        return JsonResponse({
            "status": False,
            "message": "wrong method"
        }, status=405)


def login(req):
    if req.method=="POST":
        json_data = json.loads(req.body)
        email = json_data.get("email")
        password =json_data.get("password")
        user = authenticate(email=email,password=password)
        if user is not None:
            login_user(req, user)
            return JsonResponse({
                "status": True,
                "message": "تم تسجيل الدخول بنجاح",
                "fullname": user.fullname,
                "email": user.email,
                "picture": user.picture,
                "user_id": user.id
            }
            )
        else:
            return JsonResponse({
                "status": False,
                "message": "لايوجد مستخدم بهذا البريد الإلكتروني وكلمة السر-"
            }, status=400)
    else:
        return JsonResponse({
            "status": False,
            "message": "wrong method"
        }, status=405)


def google_login(req):
    if req.method == "POST":
        json_data = json.loads(req.body)
        email = json_data.get("email")
        fullname = json_data.get("fullname")
        picture = json_data.get("picture")
        user_model = get_user_model()
        user = user_model(email=email, fullname=fullname,picture=picture)
        try:
            user.save()
        except IntegrityError:
            user = user_model.objects.get(email=email)
        login_user(req, user)
        return JsonResponse({
            "status": True,
            "message": "تم تسجيل الدخول بنجاح",
            "fullname": user.fullname,
            "email": user.email,
            "picture": user.picture,
            "user_id": user.id
        })

    else:
        return JsonResponse({
            "status": False,
            "message": "wrong method"
        }, status=405)


def logout(req):
    if req.method == "GET":
        logout_user(req)
        return JsonResponse({
            "message": "تم تسجيل الخروج بنجاح"
        })
    else:
        return JsonResponse({
            "status": False,
            "message": "wrong method"
        }, status=405)


def verify_login(req):
    if req.user.is_anonymous:
        logout_user(req)
        return JsonResponse({
            "status": False,
            "message": "user is logged out"
        }, status=401)
    else:
        return JsonResponse({
            "status": True,
            "message": "user is logged in"
        }, status=200)
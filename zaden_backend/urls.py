from django.http import JsonResponse
from django.urls import include, path

urlpatterns = [
    path('auth/',include('core.urls'))
]

def error_404(request, exception):
        return JsonResponse({
                "status": False,
                "message": "wrong URL"
        },status = 404)

def error_500(request):
        return JsonResponse({
                "status": False,
                "message": "server error"
        },status = 500)

handler404 = error_404
handler500 = error_500
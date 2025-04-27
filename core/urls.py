from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import XMDataViewSet

router = DefaultRouter()
router.register(r'xm-data', XMDataViewSet, basename='xm-data')

urlpatterns = [
    path('', include(router.urls)),
]
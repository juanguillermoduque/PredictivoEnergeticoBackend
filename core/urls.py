from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import XMDataViewSet,PredictionViewSet

router = DefaultRouter()
router.register(r'xm-data', XMDataViewSet, basename='xm-data')
router.register(r'predictions', PredictionViewSet, basename='predictions')

urlpatterns = [
    path('', include(router.urls)),
]
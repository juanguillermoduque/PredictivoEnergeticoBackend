from django.contrib import admin
from .models import XMData, Prediction

@admin.register(XMData)
class XMDataAdmin(admin.ModelAdmin):
    list_display = ('fecha', 'demanda_real', 'generacion_total', 'precio_bolsa', 'created_at')
    list_filter = ('fecha',)
    search_fields = ('fecha',)
    ordering = ('-fecha',)

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ('fecha', 'valor_predicho', 'tipo_prediccion', 'created_at')
    list_filter = ('tipo_prediccion', 'fecha')
    search_fields = ('tipo_prediccion', 'fecha')
    ordering = ('-fecha',)

from django.db import models

# Create your models here.

class XMData(models.Model):
    fecha = models.DateTimeField()
    demanda_real = models.FloatField()
    generacion_total = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-fecha']
        verbose_name = 'Dato XM'
        verbose_name_plural = 'Datos XM'

    def __str__(self):
        return f"Datos XM - {self.fecha}"

class Prediction(models.Model):
    fecha = models.DateTimeField()
    valor_predicho = models.FloatField()
    tipo_prediccion = models.CharField(max_length=50)  # demanda, generacion, precio
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-fecha']
        verbose_name = 'Predicción'
        verbose_name_plural = 'Predicciones'

    def __str__(self):
        return f"Predicción {self.tipo_prediccion} - {self.fecha}"

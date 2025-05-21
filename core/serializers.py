from rest_framework import serializers
from .models import XMData, Prediction

class XMDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = XMData
        fields = '__all__'

class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = '__all__'

class DateRangeSerializer(serializers.Serializer):
    start_date = serializers.DateTimeField()
    end_date = serializers.DateTimeField()

class PredictionRequestSerializer(serializers.Serializer):
    start_date = serializers.DateTimeField()
    end_date = serializers.DateTimeField()
    target_column = serializers.ChoiceField(choices=['demanda_real', 'generacion_total'])
    steps_ahead = serializers.IntegerField(min_value=1, max_value=168, default=24) 
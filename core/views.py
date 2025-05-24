from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.utils import timezone
from datetime import timedelta
import pandas as pd

from .models import XMData, Prediction
from .serializers import (
    XMDataSerializer,
    PredictionSerializer,
    DateRangeSerializer,
    PredictionRequestSerializer
)
from .services.xm_api import XMService
from .services.ml import MLService
from .services.visualization import VisualizationService
from .services.visualization_service import VisualizationService
from .services.prediction_service import PredictionService

class XMDataViewSet(viewsets.ModelViewSet):
    queryset = XMData.objects.all()
    serializer_class = XMDataSerializer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xm_service = XMService()
        self.visualization_service = VisualizationService()

    @action(detail=False, methods=['post'])
    def fetch_data(self, request):
        serializer = DateRangeSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        start_date = serializer.validated_data['start_date']
        end_date = serializer.validated_data['end_date']

        xm_service = XMService()
        data = xm_service.get_all_data(start_date, end_date)

        if data is None:
            return Response(
                {'error': 'No se pudieron obtener los datos'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # Guardar datos en la base de datos
        for _, row in data.iterrows():
            XMData.objects.create(
                fecha=row['fecha'],
                demanda_real=row['demanda_real'],
                generacion_total=row['generacion_total']
            )

        return Response({'message': 'Datos obtenidos y guardados correctamente'})

    @action(detail=False, methods=['get'])
    def visualizations(self, request):
        try:
            start_date = request.query_params.get('start_date')
            end_date = request.query_params.get('end_date')
            chart_type = request.query_params.get('type', 'daily')
    
            # Validar que las fechas no sean futuras
            current_date = timezone.now().date()
            if pd.to_datetime(start_date).date() > current_date or pd.to_datetime(end_date).date() > current_date:
                return Response(
                    {"error": "Cannot fetch data for future dates. Please use past dates."}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
    
            # Sugerir un rango de fechas válido
            suggested_end_date = current_date
            suggested_start_date = (current_date - timedelta(days=30)).strftime('%Y-%m-%d')
            
            if not all([start_date, end_date]):
                return Response({
                    "error": "start_date and end_date are required",
                    "suggestion": f"Try using date range: {suggested_start_date} to {suggested_end_date}"
                }, status=status.HTTP_400_BAD_REQUEST)
            
            
            # Si no hay datos en la BD, intentar obtenerlos de la API
            data = self.xm_service.get_all_data(start_date, end_date)
            if data is None:
                return Response(
                    {"error": "No data available for the specified date range"}, 
                    status=status.HTTP_404_NOT_FOUND
                )

            visualization_methods = {
                'outler': self.visualization_service.create_outliers,
                'coolwarm': self.visualization_service.create_coolwarm,
                'demanda_real_vs_z_score': self.visualization_service.create_demanda_real_vs_z_score,
                'box_plot': self.visualization_service.create_box_plot,
                'histogram': self.visualization_service.create_histograma,
                'warm_map': self.visualization_service.create_warm_map,
                'acumilativo_demanda_horaria': self.visualization_service.create_acumilativo_demanda_horaria,
                'line_demanda_promedio': self.visualization_service.create_line_demanda_promedio,
                'barras_demanda_promedio_hora': self.visualization_service.create_barras_demanda_promedio,
                'barras_demanda_promedio_mes': self.visualization_service.create_barras_demanda_promedio_mes,
                'barras_demanda_promedio_dia': self.visualization_service.create_barras_demanda_promedio_dia,
                'barras_demanda_promedio_anio': self.visualization_service.create_barras_demanda_promedio_anio,
                'barras_acumulado_promedio_anio_mes': self.visualization_service.create_barras_acumulado_promedio_anio_mes,
                'dispersion_horaria': self.visualization_service.create_dispersion_horaria,
                'historico_minimo_maximo': self.visualization_service.create_historico_minimo_maximo,

            }

            if chart_type not in visualization_methods:
                return Response(
                    {"error": f"Invalid chart type. Available types: {list(visualization_methods.keys())}"}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

            chart_data = visualization_methods[chart_type](data)
            return Response(chart_data)

        except Exception as e:
            return Response(
                {"error": f"Error processing visualization: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['get'])
    def get_visualizations(self, request):
        """
        Obtiene diferentes visualizaciones de los datos
        """
        start_date = request.query_params.get('start_date')
        end_date = request.query_params.get('end_date')
        visualization_type = request.query_params.get('type', 'time_series')

        if not start_date or not end_date:
            return Response(
                {'error': 'Se requieren fechas de inicio y fin'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Obtener datos
        queryset = self.queryset.filter(
            fecha__gte=start_date,
            fecha__lte=end_date
        ).order_by('fecha')

        if not queryset.exists():
            return Response(
                {'error': 'No hay datos para el rango de fechas especificado'},
                status=status.HTTP_404_NOT_FOUND
            )

        # Convertir a DataFrame
        data = pd.DataFrame(list(queryset.values()))

        # Crear visualización
        viz_service = VisualizationService()
        
        if visualization_type == 'time_series':
            plot_data = viz_service.create_time_series(
                data,
                'Serie Temporal de Datos XM',
                'Valor'
            )
        elif visualization_type == 'correlation':
            plot_data = viz_service.create_correlation_heatmap(data)
        elif visualization_type == 'box_plot':
            column = request.query_params.get('column', 'demanda_real')
            plot_data = viz_service.create_box_plot(data, column)
        elif visualization_type == 'histogram':
            column = request.query_params.get('column', 'demanda_real')
            plot_data = viz_service.create_histogram(data, column)
        else:
            return Response(
                {'error': 'Tipo de visualización no válido'},
                status=status.HTTP_400_BAD_REQUEST
            )

        return Response({'plot_data': plot_data})

class PredictionViewSet(viewsets.ModelViewSet):
    queryset = Prediction.objects.all()
    serializer_class = PredictionSerializer

    @action(detail=False, methods=['post'])
    def predict(self, request):
        serializer = PredictionRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        start_date = serializer.validated_data['start_date']
        end_date = serializer.validated_data['end_date']

        # Obtener datos históricos
        xm_service = XMService()
        data = xm_service.get_all_data("2022-01-01", "2024-12-31")

        if data is None:
            return Response(
                {'error': 'No se pudieron obtener los datos históricos'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
        # llamar al servicio de predicción
        prediction_service = PredictionService()
        normalized_data = prediction_service.normalize_data(data)
        prediction_service.create_modelos_dl(normalized_data)
        prediction_service.generate_models()
        base_64 =  prediction_service.predict(normalized_data, start_date, end_date)
        return Response(base_64)

       

    @action(detail=False, methods=['get'])
    def get_latest_predictions(self, request):
        tipo_prediccion = request.query_params.get('tipo', None)
        queryset = self.queryset

        if tipo_prediccion:
            queryset = queryset.filter(tipo_prediccion=tipo_prediccion)

        queryset = queryset.order_by('-fecha')[:24]
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['post'])
    def compare_models(self, request):
        """
        Compara diferentes modelos de predicción
        """
        serializer = PredictionRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        start_date = serializer.validated_data['start_date']
        end_date = serializer.validated_data['end_date']

        # Obtener datos históricos
        xm_service = XMService()
        data = xm_service.get_all_data(start_date, end_date)

        if data is None:
            return Response(
                {'error': 'No se pudieron obtener los datos históricos'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # Comparar modelos
        ml_service = MLService()
        results = ml_service.compare_models(data)

        return Response({
            'message': 'Comparación de modelos realizada correctamente',
            'results': results
        })

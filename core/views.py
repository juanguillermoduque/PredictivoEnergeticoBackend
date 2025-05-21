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
                'heatmap': self.visualization_service.create_heatmap,
                'daily': self.visualization_service.create_daily_demand,
                'hourly': self.visualization_service.create_hourly_average
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
        target_column = serializer.validated_data['target_column']
        steps_ahead = serializer.validated_data['steps_ahead']
        model_type = request.data.get('model_type', 'random_forest')

        # Obtener datos históricos
        xm_service = XMService()
        data = xm_service.get_all_data(start_date, end_date)

        if data is None:
            return Response(
                {'error': 'No se pudieron obtener los datos históricos'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # Entrenar modelo y hacer predicciones
        ml_service = MLService()
        model_name = f"{target_column}_{model_type}"
        
        try:
            # Entrenar modelo
            metrics = ml_service.train_model(data, target_column, model_name, model_type)
            
            # Realizar predicciones
            predictions = ml_service.predict(data, target_column, model_name, steps_ahead)
            
            # Crear visualización
            viz_service = VisualizationService()
            plot_data = viz_service.create_prediction_comparison(
                data,
                predictions,
                target_column
            )
            
            # Guardar predicciones
            for _, row in predictions.iterrows():
                Prediction.objects.create(
                    fecha=row['fecha'],
                    valor_predicho=row[f'prediccion_{target_column}'],
                    tipo_prediccion=target_column
                )

            return Response({
                'message': 'Predicciones realizadas correctamente',
                'metrics': metrics,
                'predictions': predictions.to_dict(orient='records'),
                'plot_data': plot_data
            })

        except Exception as e:
            return Response(
                {'error': f'Error al realizar predicciones: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['get'])
    def get_latest_predictions(self, request):
        tipo_prediccion = request.query_params.get('tipo', None)
        queryset = self.queryset

        if tipo_prediccion:
            queryset = queryset.filter(tipo_prediccion=tipo_prediccion)

        queryset = queryset.order_by('-fecha')[:24]  # Últimas 24 predicciones
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
        target_column = serializer.validated_data['target_column']
        steps_ahead = serializer.validated_data['steps_ahead']

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
        results = ml_service.compare_models(data, target_column, steps_ahead)

        return Response({
            'message': 'Comparación de modelos realizada correctamente',
            'results': results
        })

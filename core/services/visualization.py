import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import json

class VisualizationService:
    def __init__(self):
        self.colors = {
            'demanda_real': '#1f77b4',
            'generacion_total': '#2ca02c',
            'precio_bolsa': '#ff7f0e',
            'prediccion': '#d62728'
        }

    def create_time_series(self, df, title, y_axis_title):
        """
        Crea una gráfica de serie temporal
        """
        fig = go.Figure()
        
        for column in df.columns:
            if column != 'fecha':
                fig.add_trace(go.Scatter(
                    x=df['fecha'],
                    y=df[column],
                    name=column,
                    line=dict(color=self.colors.get(column, '#1f77b4'))
                ))

        fig.update_layout(
            title=title,
            xaxis_title='Fecha',
            yaxis_title=y_axis_title,
            template='plotly_white',
            hovermode='x unified'
        )

        return json.dumps(fig.to_dict())

    def create_prediction_comparison(self, historical_data, predictions, target_column):
        """
        Crea una gráfica comparando datos históricos con predicciones
        """
        fig = go.Figure()

        # Datos históricos
        fig.add_trace(go.Scatter(
            x=historical_data['fecha'],
            y=historical_data[target_column],
            name='Datos Históricos',
            line=dict(color=self.colors[target_column])
        ))

        # Predicciones
        fig.add_trace(go.Scatter(
            x=predictions['fecha'],
            y=predictions[f'prediccion_{target_column}'],
            name='Predicciones',
            line=dict(color=self.colors['prediccion'], dash='dash')
        ))

        fig.update_layout(
            title=f'Predicción de {target_column}',
            xaxis_title='Fecha',
            yaxis_title=target_column,
            template='plotly_white',
            hovermode='x unified'
        )

        return json.dumps(fig.to_dict())

    def create_correlation_heatmap(self, df):
        """
        Crea un mapa de calor de correlaciones
        """
        corr_matrix = df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))

        fig.update_layout(
            title='Matriz de Correlación',
            template='plotly_white'
        )

        return json.dumps(fig.to_dict())

    def create_box_plot(self, df, column):
        """
        Crea un gráfico de caja para una columna específica
        """
        fig = go.Figure()

        fig.add_trace(go.Box(
            y=df[column],
            name=column,
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ))

        fig.update_layout(
            title=f'Distribución de {column}',
            yaxis_title=column,
            template='plotly_white'
        )

        return json.dumps(fig.to_dict())

    def create_histogram(self, df, column):
        """
        Crea un histograma para una columna específica
        """
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=df[column],
            name=column,
            nbinsx=30
        ))

        fig.update_layout(
            title=f'Distribución de {column}',
            xaxis_title=column,
            yaxis_title='Frecuencia',
            template='plotly_white'
        )

        return json.dumps(fig.to_dict())

    def create_forecast_plot(self, historical_data, predictions, target_column, confidence_interval=None):
        """
        Crea una gráfica de pronóstico con intervalo de confianza
        """
        fig = go.Figure()

        # Datos históricos
        fig.add_trace(go.Scatter(
            x=historical_data['fecha'],
            y=historical_data[target_column],
            name='Datos Históricos',
            line=dict(color=self.colors[target_column])
        ))

        # Predicciones
        fig.add_trace(go.Scatter(
            x=predictions['fecha'],
            y=predictions[f'prediccion_{target_column}'],
            name='Predicciones',
            line=dict(color=self.colors['prediccion'])
        ))

        # Intervalo de confianza si está disponible
        if confidence_interval is not None:
            fig.add_trace(go.Scatter(
                x=predictions['fecha'],
                y=confidence_interval['upper'],
                fill=None,
                mode='lines',
                line_color='rgba(0,100,80,0.2)',
                name='Intervalo Superior'
            ))
            fig.add_trace(go.Scatter(
                x=predictions['fecha'],
                y=confidence_interval['lower'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,100,80,0.2)',
                name='Intervalo Inferior'
            ))

        fig.update_layout(
            title=f'Pronóstico de {target_column}',
            xaxis_title='Fecha',
            yaxis_title=target_column,
            template='plotly_white',
            hovermode='x unified'
        )

        return json.dumps(fig.to_dict()) 
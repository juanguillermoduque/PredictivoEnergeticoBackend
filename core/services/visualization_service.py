import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

class VisualizationService:
    def create_heatmap(self, df):
        hourly_columns = [col for col in df.columns if col.startswith('Values_Hour')]
        correlation_matrix = df[hourly_columns].corr()
        
        return {
            'z': correlation_matrix.values.tolist(),
            'x': hourly_columns,
            'y': hourly_columns,
            'type': 'heatmap',
            'colorscale': 'Viridis'
        }

    def create_daily_demand(self, df):
        daily_demand = df.groupby('fecha')['demanda_real'].sum().reset_index()
        
        return {
            'x': daily_demand['fecha'].tolist(),
            'y': daily_demand['demanda_real'].tolist(),
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': 'Demanda Diaria'
        }

    def create_hourly_average(self, df):
        hourly_avg = df[[col for col in df.columns if col.startswith('Values_Hour')]].mean()
        
        return {
            'x': list(range(1, 25)),
            'y': hourly_avg.values.tolist(),
            'type': 'bar',
            'name': 'Promedio por Hora'
        }

    def create_monthly_analysis(self, df):
        df['mes'] = pd.to_datetime(df['fecha']).dt.month
        monthly_avg = df.groupby('mes')['demanda_real'].mean().reset_index()
        
        return {
            'x': monthly_avg['mes'].tolist(),
            'y': monthly_avg['demanda_real'].tolist(),
            'type': 'bar',
            'name': 'Promedio Mensual'
        }

    def create_weekly_analysis(self, df):
        df['semana'] = pd.to_datetime(df['fecha']).dt.isocalendar().week
        weekly_avg = df.groupby('semana')['demanda_real'].mean().reset_index()
        
        return {
            'x': weekly_avg['semana'].tolist(),
            'y': weekly_avg['demanda_real'].tolist(),
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': 'Promedio Semanal'
        }

    def create_yearly_analysis(self, df):
        df['año'] = pd.to_datetime(df['fecha']).dt.year
        yearly_stats = df.groupby('año').agg({
            'demanda_real': ['mean', 'sum', 'std']
        }).reset_index()
        
        return {
            'yearly_mean': {
                'x': yearly_stats['año'].tolist(),
                'y': yearly_stats['demanda_real']['mean'].tolist(),
                'type': 'bar',
                'name': 'Promedio Anual'
            },
            'yearly_total': {
                'x': yearly_stats['año'].tolist(),
                'y': yearly_stats['demanda_real']['sum'].tolist(),
                'type': 'bar',
                'name': 'Total Anual'
            }
        }

    def create_outliers_analysis(self, df):
        z_scores = stats.zscore(df['demanda_real'])
        outliers = (abs(z_scores) > 3)
        
        return {
            'normal': {
                'x': df[~outliers]['fecha'].tolist(),
                'y': df[~outliers]['demanda_real'].tolist(),
                'type': 'scatter',
                'mode': 'markers',
                'name': 'Datos Normales'
            },
            'outliers': {
                'x': df[outliers]['fecha'].tolist(),
                'y': df[outliers]['demanda_real'].tolist(),
                'type': 'scatter',
                'mode': 'markers',
                'name': 'Outliers',
                'marker': {'color': 'red'}
            }
        }

    def create_distribution_analysis(self, df):
        return {
            'histogram': {
                'x': df['demanda_real'].tolist(),
                'type': 'histogram',
                'nbinsx': 50,
                'name': 'Distribución de Demanda'
            },
            'box': {
                'y': df['demanda_real'].tolist(),
                'type': 'box',
                'name': 'Box Plot Demanda'
            }
        }

    def create_time_series_decomposition(self, df):
        df['fecha'] = pd.to_datetime(df['fecha'])
        df = df.set_index('fecha')
        decomposition = stats.seasonal_decompose(
            df['demanda_real'], 
            period=24
        )
        
        return {
            'trend': {
                'x': decomposition.trend.index.tolist(),
                'y': decomposition.trend.values.tolist(),
                'type': 'scatter',
                'name': 'Tendencia'
            },
            'seasonal': {
                'x': decomposition.seasonal.index.tolist(),
                'y': decomposition.seasonal.values.tolist(),
                'type': 'scatter',
                'name': 'Estacionalidad'
            },
            'residual': {
                'x': decomposition.resid.index.tolist(),
                'y': decomposition.resid.values.tolist(),
                'type': 'scatter',
                'name': 'Residuos'
            }
        }

    def get_statistical_summary(self, df):
        return {
            'basic_stats': df['demanda_real'].describe().to_dict(),
            'correlation_matrix': df[[col for col in df.columns 
                                   if col.startswith('Values_Hour')]].corr().to_dict(),
            'hourly_stats': df[[col for col in df.columns 
                              if col.startswith('Values_Hour')]].describe().to_dict()
        }
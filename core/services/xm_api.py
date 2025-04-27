from pydataxm.pydataxm import ReadDB
import pandas as pd
from datetime import datetime

class XMService:
    def __init__(self):
        self.api = ReadDB()

    def get_demand_data(self, start_date, end_date):
        """
        Obtiene datos de demanda real del sistema usando pydataxm
        """
        try:
            df = self.api.request_data('DemaReal', 'Sistema', start_date, end_date)
            # Aseguramos formato y columnas
            df = df.rename(columns={'Date': 'fecha'})
            df = df[['fecha'] + [col for col in df.columns if col.startswith('Values_Hour')]]
            # Sumar todas las horas para demanda total diaria
            df['demanda_real'] = df[[col for col in df.columns if col.startswith('Values_Hour')]].sum(axis=1)
            df = df[['fecha', 'demanda_real']]
            return df
        except Exception as e:
            print(f"Error al obtener datos de demanda: {e}")
            return None

    def get_generation_data(self, start_date, end_date):
        """
        Obtiene datos de generación por recurso usando pydataxm
        """
        try:
            df = self.api.request_data('Gene', 'Recurso', start_date, end_date)
            df = df.rename(columns={'Date': 'fecha'})
            df = df[['fecha'] + [col for col in df.columns if col.startswith('Values_Hour')]]
            df['generacion_total'] = df[[col for col in df.columns if col.startswith('Values_Hour')]].sum(axis=1)
            df = df[['fecha', 'generacion_total']]
            return df
        except Exception as e:
            print(f"Error al obtener datos de generación: {e}")
            return None

    def get_price_data(self, start_date, end_date):
        """
        Obtiene datos de precio de bolsa usando pydataxm (ajustar si hay colección específica)
        """
        try:
            df = self.api.request_data('PrecioBolsa', 'Sistema', start_date, end_date)
            df = df.rename(columns={'Date': 'fecha', 'Values_PrecioBolsa': 'precio_bolsa'})
            df = df[['fecha', 'precio_bolsa']]
            return df
        except Exception as e:
            print(f"Error al obtener datos de precio: {e}")
            return None

    def get_all_data(self, start_date, end_date):
        """
        Obtiene todos los datos necesarios para el análisis usando pydataxm
        """
        demand_df = self.get_demand_data(start_date, end_date)
        generation_df = self.get_generation_data(start_date, end_date)
        price_df = self.get_price_data(start_date, end_date)

        if all(df is not None for df in [demand_df, generation_df, price_df]):
            final_df = demand_df.merge(generation_df, on='fecha', how='outer')
            final_df = final_df.merge(price_df, on='fecha', how='outer')
            return final_df
        return None
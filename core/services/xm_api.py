from pydataxm.pydataxm import ReadDB
import pandas as pd
import numpy as np
import datetime as dt
import calendar
import json
import matplotlib.pyplot as plt
import seaborn as sns 

class XMService:
    def __init__(self):
        self.api = ReadDB()

    def custom_json_encoder(self,obj):
        # Convertir tipos numpy a tipos estándar de Python
        if isinstance(obj, np.int64):  # Si es un tipo numpy int64
            return int(obj)  # Convertir a int
        elif isinstance(obj, np.float64):  # Si es un tipo numpy float64
            return float(obj)  # Convertir a float
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()  # Convertir el DataFrame a un diccionario
        elif isinstance(obj, pd.Series):
            return obj.to_dict()  # Convertir la Serie a un diccionario
        elif isinstance(obj, pd.Timestamp):  # Si es un tipo pandas Timestamp
            return obj.isoformat()  # Convertir a una cadena ISO 8601
        elif isinstance(obj, (int, float, str, list, dict, type)):
            return obj  # Los valores básicos son serializables
        else:
            raise TypeError(f"Tipo no serializable: {type(obj)}")  # Generar un error para tipos no serializables
    
    def get_demand_data(self, start_date, end_date):
        """
        Obtiene datos de demanda real del sistema usando pydataxm
        """
        try:    
            #Se pasa la fecha de string a datetime
     
            year, month, day = map(int, start_date.split('-'))
            start_date = dt.date(year, month, day)
            year, month, day = map(int, end_date.split('-'))
            end_date = dt.date(year, month, day)

            df_DemaReal_sistema = {}      
            current_date = start_date
            # Iniciamos el ciclo que itera desde la fecha actual hasta la fecha final
            while current_date <= end_date:
            
                # Calcular el último día del mes actual. Utilizamos calendar.monthrange() para obtener el número de días en el mes.
                last_day_of_month = dt.date(current_date.year, current_date.month, calendar.monthrange(current_date.year, current_date.month)[1])
                
                # Verificamos si el último día del mes es posterior al 30 de noviembre de 2024
                # Si es así, ajustamos la fecha al 30 de noviembre de 2024
                if last_day_of_month > end_date:
                    last_day_of_month = end_date
                
                # Realizamos una solicitud de datos a la API para obtener los registros del mes actual
                # Suponemos que la función 'request_data' retorna un DataFrame con los datos
                df_dema_real = self.api.request_data('DemaReal', 'Sistema', current_date, last_day_of_month)
                
                # Comprobamos si la API devuelve datos y si el DataFrame no está vacío
                if df_dema_real is not None and not df_dema_real.empty:
                    # Extraemos el nombre del mes en formato completo (por ejemplo, "January", "February")
                    mes_nombre = current_date.strftime('%B_%Y')  # Usa el nombre completo del mes
                    
                    # Almacenamos los datos obtenidos en un diccionario, usando el nombre del mes como clave
                    # Si ya existe una entrada para el mes, concatenamos los nuevos datos al DataFrame existente
                    if mes_nombre in df_DemaReal_sistema:
                        df_DemaReal_sistema[mes_nombre] = pd.concat([df_DemaReal_sistema[mes_nombre], df_dema_real])
                    else:
                        # Si no existe una entrada para el mes, la creamos con los datos obtenidos
                        df_DemaReal_sistema[mes_nombre] = df_dema_real

                # Avanzamos al primer día del siguiente mes
                # Si estamos en diciembre, saltamos al primer día de enero del siguiente año
                if current_date.month == 12:
                    current_date = dt.date(current_date.year + 1, 1, 1)
                else:
                    # Si no estamos en diciembre, simplemente incrementamos el mes
                    current_date = dt.date(current_date.year, current_date.month + 1, 1)
            return df_DemaReal_sistema
        except Exception as e:
            print(f"Error al obtener datos de demanda: {e}")
            return None
        
    def get_demand_data_final(self,df_DemaReal_sistema):
        '''
        Obtiene datos de demanda real del sistema usando pydataxm
        '''
        try:
            df_DemaReal_sistema_final = pd.concat(df_DemaReal_sistema.values(), ignore_index=True)
            # Organizar las columnas
            df_DemaReal_sistema_final = df_DemaReal_sistema_final[['Date', 'Values_Hour01', 'Values_Hour02', 'Values_Hour03', 'Values_Hour04', 'Values_Hour05', 'Values_Hour06', 'Values_Hour07', 'Values_Hour08', 'Values_Hour09', 'Values_Hour10', 'Values_Hour11', 'Values_Hour12', 'Values_Hour13', 'Values_Hour14', 'Values_Hour15', 'Values_Hour16', 'Values_Hour17', 'Values_Hour18', 'Values_Hour19', 'Values_Hour20', 'Values_Hour21', 'Values_Hour22', 'Values_Hour23', 'Values_Hour24']]

            # Obtener el timestamp actual
            timestamp = dt.datetime.now().strftime("%Y_%m_%d")

            # Guardar el DataFrame en un archivo CSV
            df_DemaReal_sistema_final.to_csv(f'DemaReal_sistema_R{timestamp}.csv', index=True)

            # Guardar en un archivo JSON de dataframe de variables por filas
            with open(f"DemaReal_sistema_2023_2024__R{timestamp}.json", "w") as file:
                json.dump(df_DemaReal_sistema_final.to_dict(orient='records'), file, default=self.custom_json_encoder, indent=4)

            # Asegúrate de que la columna 'Date' esté en formato datetime
            df_DemaReal_sistema_final['Date'] = pd.to_datetime(df_DemaReal_sistema_final['Date'])

            # Ordenar el DataFrame por la columna 'Date'
            df_DemaReal_sistema_final = df_DemaReal_sistema_final.sort_values(by='Date')
            return df_DemaReal_sistema_final
        except Exception as e:
            print(f"Error al obtener datos de demanda: {e}")
            return None

    def get_all_data(self, start_date, end_date):
        """
        Obtiene todos los datos necesarios para el análisis usando pydataxm
        """
        demand_df = self.get_demand_data(start_date, end_date)
        demand_df_final = self.get_demand_data_final(demand_df)
        
        if all(df is not None for df in [demand_df, demand_df_final]):
            return demand_df_final
        return None
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from scipy.stats import zscore
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU, Conv1D, MaxPooling1D  
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import plotly.graph_objects as go
import base64
from io import BytesIO
from tensorflow.keras.models import load_model
import os

class PredictionService: 
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = None
        self.sequence_length = 24
        self.X_train, self.X_test, self.y_train, self.y_test,self.early_stop,self.seq_length = None, None, None, None, None, None
        self.model_1, self.model_2, self.model_3, self.model_4, self.model_5 = None, None, None, None, None
        self.predictions_1, self.predictions_2, self.predictions_3, self.predictions_4, self.predictions_5 = None, None, None, None, None

        # Escalador para las horas
        self.hour_scaler = MinMaxScaler(feature_range=(0, 1))

        # Escalador para Total_kWh
        self.total_kWh_scaler = StandardScaler()

        # Normalizar la columna 'Month' con LabelEncoder
        self.month_encoder = LabelEncoder()

        # Normalizar la columna 'Weekday' con LabelEncoder
        self.weekday_encoder = LabelEncoder()

        # Normalizar la columna 'Week' con LabelEncoder
        self.week_encoder = LabelEncoder()

        # Normalizar la columna 'Year' con LabelEncoder
        self.holiday_encoder = LabelEncoder()

        # Noemalizar la columna 'Year' con LabelEncoder
        self.year_encoder = LabelEncoder()

    def normalize_data(self, df):
        df['Total_kWh'] = df.select_dtypes(include='number').sum(axis=1)
        df['z_score'] = zscore(df['Total_kWh'])
        df_DemaReal_sistema_final = df
        # Normalizamos la columna Total_kWh con StandardScaler (esto usa media y desviación estándar)
        df_DemaReal_sistema_final['Total_kWh'] = self.total_kWh_scaler.fit_transform(df_DemaReal_sistema_final[['Total_kWh']])

        # Normalizamos las columnas horarias con MinMaxScaler
        hour_columns = [f'Values_Hour{str(i).zfill(2)}' for i in range(1, 25)]  # Genera la lista de las horas
        for hour in hour_columns:
            df_DemaReal_sistema_final[hour] = self.hour_scaler.fit_transform(df_DemaReal_sistema_final[[hour]])

        # Agregar dia de la semana como columna categorica
        df_DemaReal_sistema_final['Weekday'] = df_DemaReal_sistema_final['Date'].dt.day_name()
        # Normalizar la columna 'Weekday' con LabelEncoder
        df_DemaReal_sistema_final['Weekday'] = self.weekday_encoder.fit_transform(df_DemaReal_sistema_final['Weekday']) + 1  # Sumar 1 para evitar el cero

        # Crear columna del mes como columna categorica
        df_DemaReal_sistema_final['Month'] = df_DemaReal_sistema_final['Date'].dt.month
        # Normalizar la columna 'Month' con LabelEncoder
        df_DemaReal_sistema_final['Month'] = self.month_encoder.fit_transform(df_DemaReal_sistema_final['Month']) + 1  # Sumar 1 para evitar el cero

        # Agregar columna numero de semnana
        df_DemaReal_sistema_final['Week'] = df_DemaReal_sistema_final['Date'].dt.isocalendar().week
        # Normalizar la columna 'Week' con LabelEncoder
        df_DemaReal_sistema_final['Week'] = self.week_encoder.fit_transform(df_DemaReal_sistema_final['Week']) + 1  # Sumar 1 para evitar el cero

        # Agregar columna de dias de lunes a viernes con indicador de 1 y fin de semana con indicador de 2
        df_DemaReal_sistema_final['Weekend'] = 1
        df_DemaReal_sistema_final.loc[df_DemaReal_sistema_final['Date'].dt.dayofweek >= 5, 'Weekend'] = 2

        # Agregar columna del dia domingo con un 1  y el resto de dias con 2
        df_DemaReal_sistema_final['Sunday'] = 2
        df_DemaReal_sistema_final.loc[df_DemaReal_sistema_final['Date'].dt.dayofweek == 6, 'Sunday'] = 1

        # Normalizar la columna 'Year' con LabelEncoder
        df_DemaReal_sistema_final['Year'] = self.year_encoder.fit_transform(df_DemaReal_sistema_final['Date'].dt.year) + 1

        # Agregar columna por dia festivo
        df_DemaReal_sistema_final['Holiday'] = 1
        # Verificar si la fecha es un día festivo
        df_DemaReal_sistema_final.loc[df_DemaReal_sistema_final['Date'].isin(['2022-1-1', '2022-1-10', '2022-3-21', '2022-4-14', '2022-4-15', '2022-5-1', '2022-5-30', '2022-6-20', '2022-6-27', '2022-7-4', '2022-7-20', '2022-8-7', '2022-8-15', '2022-10-17', '2022-11-7', '2022-11-1', '2022-12-8', '2022-12-25', '2023-1-1', '2023-1-9', '2023-3-20', '2023-4-6', '2023-4-7', '2023-5-1', '2023-5-22', '2023-6-12', '2023-6-19', '2023-7-3', '2023-7-20', '2023-8-7', '2023-8-21', '2023-10-16', '2023-11-06', '2023-11-13', '2023-12-08', '2023-12-25', '2024-1-1', '2024-1-8', '2024-3-25', '2024-3-28', '2024-3-29', '2024-5-1', '2024-5-13', '2024-6-3', '2024-6-10', '2024-7-1', '2024-7-20', '2024-8-7', '2024-8-19', '2024-10-14', '2024-11-4', '2024-11-11', '2024-12-08', '2024-12-25']), 'Holiday'] = 2
        # Normalizar la columna 'Holiday' con LabelEncoder
        df_DemaReal_sistema_final['Holiday'] = self.holiday_encoder.fit_transform(df_DemaReal_sistema_final['Holiday']) + 1  # Sumar 1 para evitar el cero

        return df_DemaReal_sistema_final

    def create_sequences(self,data, seq_length):
        """
        Genera secuencias de datos para el entrenamiento de modelos de predicción.
        """
        xs, ys = [], []  # Inicialización de listas para las secuencias de entrada y los valores objetivo
        
        # Generación de nombres de columnas de horas
        hour_columns = [f'Values_Hour{str(i).zfill(2)}' for i in range(1, 25)]
        
        # Recorremos el DataFrame desde 'seq_length' hasta el final
        for i in range(seq_length, len(data)):  
            # Obtenemos las características de las horas y otras características adicionales
            x_hours = data.iloc[i-seq_length:i][hour_columns].values
            x_extra = data.iloc[i-seq_length:i][['Weekday', 'Month', 'Week', 'Year', 'Holiday', 'Weekend', 'Sunday']].values
            
            # Concatenamos las horas con las características adicionales
            x = np.concatenate([x_hours, x_extra], axis=1)
            
            # El valor objetivo es el 'Total_kWh' del día siguiente
            y = data.iloc[i]['Total_kWh']
            
            # Añadimos la secuencia y el objetivo a las listas
            xs.append(x)
            ys.append(y)
        
        return np.array(xs), np.array(ys)

    def create_modelos_dl(self, df):
        # Ajuste inicial de los datos
        self.seq_length = 90  # Número de días previos a considerar para predecir el siguiente valor (6 semanas)
        X, y = self.create_sequences(df, self.seq_length)

        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

        self.X_train = X_train.astype('float32')
        self.X_test = X_test.astype('float32')
        self.y_train = y_train.astype('float32')
        self.y_test = y_test.astype('float32')

        # Definir el EarlyStopping callback
        self.early_stop = EarlyStopping(monitor='val_loss',  # Monitoreamos la pérdida en el conjunto de validación
                                patience=8,  # Número de épocas sin mejora antes de detener el entrenamiento
                                verbose=1,  # Mostrar mensajes de parada
                                restore_best_weights=True)  # Restaurar los pesos del modelo con la mejor pérdida de validación
   
    def generate_models(self):
        self.generate_model_1()
        self.generate_model_2()
        self.generate_model_3()
        self.generate_model_4()
        self.generate_model_5()

    def generate_model_1(self):
        if os.path.exists("modelo1_lstm.h5"):
            self.model_1 = load_model("modelo1_lstm.h5")
            return
        else:
            # Modelo 1: LSTM Básico con 2 capas LSTM
            model_1 = Sequential()
            model_1.add(LSTM(128, return_sequences=True, input_shape=(self.seq_length, 31), activation='tanh'))
            model_1.add(Dropout(0.2))  # Evita el sobreajuste
            model_1.add(LSTM(64, return_sequences=False, activation='tanh'))
            model_1.add(Dropout(0.2))  # Evita el sobreajuste
            model_1.add(Dense(25, activation='tanh'))
            model_1.add(Dense(1, activation='linear'))
            model_1.summary()
            model_1.compile(optimizer='adam', loss='mean_squared_error')

            # Entrenar el modelo 1
            model_1.fit(self.X_train, self.y_train, batch_size=4, epochs=25, validation_data=(self.X_test, self.y_test), callbacks=[self.early_stop])
            self.model_1 = model_1
            self.model_1.save("modelo1_lstm.h5")

    def generate_model_2(self):
        if os.path.exists("modelo2_lstm.h5"):
            self.model_2 = load_model("modelo2_lstm.h5")
            return
        else:
            # Modelo 2: LSTM con 3 capas LSTM y más Dropout
            model_2 = Sequential()
            model_2.add(LSTM(128, return_sequences=True, input_shape=(self.seq_length, 31), activation='tanh'))
            model_2.add(Dropout(0.3))  # Más Dropout
            model_2.add(LSTM(64, return_sequences=True, activation='tanh'))
            model_2.add(Dropout(0.3))  # Más Dropout
            model_2.add(LSTM(32, return_sequences=False, activation='tanh'))
            model_2.add(Dropout(0.3))  # Más Dropout
            model_2.add(Dense(20, activation='tanh'))
            model_2.add(Dense(1, activation='linear'))

            model_2.summary()

            model_2.compile(optimizer='adam', loss='mean_squared_error')

            # Entrenar el modelo 2
            model_2.fit(self.X_train, self.y_train, batch_size=8, epochs=25, validation_data=(self.X_test, self.y_test), callbacks=[self.early_stop])
            self.model_2 = model_2
            self.model_2.save("modelo2_lstm.h5")
    
    def generate_model_3(self):
        if os.path.exists("modelo3_lstm.h5"):
            self.model_3 = load_model("modelo3_lstm.h5")
            return
        else:
            # Modelo 3: LSTM Bidireccional
            model_3 = Sequential()
            model_3.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(self.seq_length, 31)))
            model_3.add(Dropout(0.2))  # Evita el sobreajuste
            model_3.add(Bidirectional(LSTM(64, return_sequences=False)))
            model_3.add(Dropout(0.2))  # Evita el sobreajuste
            model_3.add(Dense(30, activation='tanh'))
            model_3.add(Dense(1, activation='linear'))

            # Imprimir la estructura del Modelo 3
            model_3.summary()

            model_3.compile(optimizer='adam', loss='mean_squared_error')

            # Entrenar el modelo 3
            model_3.fit(self.X_train, self.y_train, batch_size=4, epochs=25, validation_data=(self.X_test, self.y_test), callbacks=[self.early_stop])

            self.model_3 = model_3
            self.model_3.save("modelo3_lstm.h5")
    
    def generate_model_4(self):
        if os.path.exists("modelo4_lstm.h5"):
            self.model_4 = load_model("modelo4_lstm.h5")
            return
        else:
            # Modelo 4: LSTM seguido de GRU
            model_4 = Sequential()
            model_4.add(LSTM(128, return_sequences=True, input_shape=(self.seq_length, 31), activation='tanh'))
            model_4.add(Dropout(0.15))  # Evita el sobreajuste
            model_4.add(GRU(64, return_sequences=False))
            model_4.add(Dropout(0.1))  # Evita el sobreajuste
            model_4.add(Dense(30, activation='tanh'))
            model_4.add(Dense(1, activation='linear'))

            # Imprimir la estructura del Modelo 4
            model_4.summary()

            model_4.compile(optimizer='adam', loss='mean_squared_error')

            # Entrenar el modelo 4
            model_4.fit(self.X_train, self.y_train, batch_size=4, epochs=25, validation_data=(self.X_test, self.y_test), callbacks=[self.early_stop])

            self.model_4 = model_4
            self.model_4.save("modelo4_lstm.h5")
    
    def generate_model_5(self):
        if os.path.exists("modelo5_lstm.h5"):
            self.model_5 = load_model("modelo5_lstm.h5")
            return
        else:
            # Modelo 5: LSTM con una capa convolucional
            model_5 = Sequential()
            model_5.add(Conv1D(64, 3, activation='relu', input_shape=(self.seq_length, 31)))
            model_5.add(MaxPooling1D(2))
            model_5.add(LSTM(128, return_sequences=True))
            model_5.add(Dropout(0.2))  # Evita el sobreajuste
            model_5.add(LSTM(64, return_sequences=False))
            model_5.add(Dropout(0.2))  # Evita el sobreajuste
            model_5.add(Dense(25, activation='tanh'))
            model_5.add(Dense(1, activation='linear'))
            model_5.summary()

            model_5.compile(optimizer='adam', loss='mean_squared_error')

            # Entrenar el modelo 5
            model_5.fit(self.X_train, self.y_train, batch_size=4, epochs=25, validation_data=(self.X_test, self.y_test), callbacks=[self.early_stop])
            self.model_5 = model_5
            self.model_5.save("modelo5_lstm.h5")

    def predict(self, df, start_date_string, end_date_string):
        df_DemaReal_sistema_final = df

        # Definir el rango de fechas para predecir (30 de noviembre de 2024 hasta el 31 de diciembre de 2024)
        start_date = pd.to_datetime(start_date_string)  # Comenzar la predicción desde el 30 de noviembre
        end_date = pd.to_datetime(end_date_string)  # Terminar la predicción el 31 de diciembre
        prediction_dates = pd.date_range(start=start_date, end=end_date)

        # Lista para almacenar las predicciones y las fechas
        dates_list = []

        # Obtener el índice de la última fila
        last_index = len(df_DemaReal_sistema_final) - 1

        # Asegurarse de que no estamos fuera de los límites
        start_index = max(last_index - self.seq_length, 0)  # No dejar que el índice sea menor que 0

        # Extraer la secuencia de 24 horas de las últimas 24 horas disponibles
        X_last_day_hours = df_DemaReal_sistema_final.iloc[start_index:start_index+self.seq_length][['Values_Hour01', 'Values_Hour02', 'Values_Hour03', 'Values_Hour04', 'Values_Hour05', 'Values_Hour06', 'Values_Hour07', 'Values_Hour08', 'Values_Hour09', 'Values_Hour10', 'Values_Hour11', 'Values_Hour12', 'Values_Hour13', 'Values_Hour14', 'Values_Hour15', 'Values_Hour16', 'Values_Hour17', 'Values_Hour18', 'Values_Hour19', 'Values_Hour20', 'Values_Hour21', 'Values_Hour22', 'Values_Hour23', 'Values_Hour24']].values

        # Extraer las columnas adicionales (Weekday, Month, Week, Holiday)
        X_last_day_extra = df_DemaReal_sistema_final.iloc[start_index:start_index+self.seq_length][['Weekday', 'Month', 'Week', 'Year', 'Holiday', 'Weekend', 'Sunday']].values

        # Concatenar las horas con las columnas adicionales
        X_last_day = np.concatenate([X_last_day_hours, X_last_day_extra], axis=1)

        # Redimensionar para LSTM
        X_last_day = X_last_day.reshape(1, self.seq_length, 31)

        # Diccionario para almacenar las predicciones de cada modelo
        models_predictions = {}

        # Definir los 5 modelos entrenados (ya entrenados previamente)
        # Asegúrate de que estas variables contengan los modelos LSTM entrenados, por ejemplo:
        models = [self.model_1, self.model_2, self.model_3, self.model_4, self.model_5]  

        # Evaluar los 5 modelos LSTM diferentes
        for model_index, model in enumerate(models):
            
            # Realizar las predicciones para los días de noviembre y diciembre de 2024
            model_predictions = []
            dates_list = []  # Reiniciar la lista de fechas para cada modelo
            
            for day in prediction_dates:
                # Realizar la predicción usando el modelo LSTM
                prediction_day = model.predict(X_last_day)  # Realizar la predicción
                predicted_value = prediction_day[0][0]  # Extraer el valor de la predicción

                # Desnormalizar la predicción si se usó un scaler para normalizar los datos
                predicted_value_desnormalized = self.total_kWh_scaler.inverse_transform([[predicted_value]])[0][0]

                # Guardar las predicciones y las fechas
                model_predictions.append(predicted_value_desnormalized)  # Guardar la predicción desnormalizada
                dates_list.append(day.date())  # Guardar la fecha del día

                # Actualizar la secuencia de datos para el siguiente día (desplazamiento temporal)
                X_last_day = np.roll(X_last_day, shift=-1, axis=1)  # Desplazar las horas hacia la izquierda
                X_last_day[0, -1, -1] = predicted_value  # Actualizar la última hora con la predicción

            for date, prediction in zip(dates_list, model_predictions):
                print(f"{date}: {prediction}")

            # Guardar las predicciones en el diccionario para cada modelo
            models_predictions[model_index + 1] = model_predictions  # Almacenar las predicciones del modelo en el diccionario

        df_DemaReal_sistema_final['Total_kWh'] = self.total_kWh_scaler.inverse_transform(df_DemaReal_sistema_final['Total_kWh'].values.reshape(-1, 1))

        # Filtrar los datos históricos hasta el 30 de noviembre de 2024
        historical_data = df_DemaReal_sistema_final[df_DemaReal_sistema_final['Date'] <= '2024-11-30']

        # Crear un gráfico interactivo con Plotly
        fig = go.Figure()

        # Agregar los datos históricos al gráfico
        fig.add_trace(go.Scatter(
            x=historical_data['Date'],
            y=historical_data['Total_kWh'],  # Suponiendo que la columna de demanda es 'Total_kWh'
            mode='lines',
            name='Datos Históricos',
            line=dict(color='blue')
        ))

        # Aquí asumimos que 'models_predictions' ya tiene las predicciones de los 5 modelos
        # Las predicciones se almacenan como models_predictions[1], models_predictions[2], ..., models_predictions[5]

        # Colores para cada modelo
        colors = ['red', 'green', 'orange', 'purple', 'brown']

        # Agregar las predicciones de cada modelo al gráfico
        for i in range(1, 6):  # Para cada uno de los 5 modelos
            model_predictions = models_predictions.get(i, [])  # Obtener las predicciones del modelo i
            if model_predictions:  # Verifica si existen predicciones para el modelo actual
                # Utilizar las fechas de predicción correspondientes para cada modelo (asumiendo que las fechas están alineadas con las predicciones)
                fig.add_trace(go.Scatter(
                    x=prediction_dates,  # Utilizar las fechas de predicción correctas
                    y=model_predictions,
                    mode='lines+markers',
                    name=f'Predicción Modelo {i}',
                    line=dict(color=colors[i-1])
                ))

        # Añadir detalles al gráfico
        fig.update_layout(
            title="Predicción de la Demanda Total_kWh ",
            xaxis_title="Fecha",
            yaxis_title="Demanda Total (kWh)",
            template='plotly_white',
            font=dict(size=13),
            xaxis=dict(
                tickformat="%d-%m-%Y",
                range=[start_date_string, end_date_string],
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=7, label="1 Semana", step="day", stepmode="backward"),
                        dict(count=1, label="1 Mes", step="month", stepmode="backward"),
                        dict(count=3, label="3 Meses", step="month", stepmode="backward"),
                        dict(step="all", label="Todo")
                    ])
                )
            )
        )

        base_64 = self.guardar_grafico_base64(fig, filename="prediccion_demanda", format="png")

        return {
            'base_64': base_64
        }

    def guardar_grafico_base64(self, fig, filename="grafico", format="png"):
        """
        Guarda la figura (matplotlib o plotly) y la retorna como base64.
        """
        try:
            buffer = BytesIO()
            # Detectar tipo de figura
            if hasattr(fig, 'savefig'):  # matplotlib
                fig.savefig(buffer, format=format)
            elif hasattr(fig, 'write_image'):  # plotly
                fig.write_image(buffer, format=format)
            else:
                raise ValueError("Tipo de figura no soportado")

            buffer.seek(0)
            base_64 = base64.b64encode(buffer.read()).decode('utf-8')
            buffer.close()
            return base_64

        except Exception as e:
            print(f"Error al guardar el gráfico: {e}")
            return None
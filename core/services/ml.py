import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from datetime import datetime, timedelta

class MLService:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_path = os.path.join(os.path.dirname(__file__), '../../models')
        os.makedirs(self.model_path, exist_ok=True)
        
        # Definir los modelos disponibles
        self.available_models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression(),
            'svr': SVR(kernel='rbf')
        }

    def prepare_data(self, df, target_column, sequence_length=24):
        """
        Prepara los datos para el modelo de predicción
        """
        # Normalizar datos
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[[target_column]])
        
        # Crear secuencias
        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:(i + sequence_length)])
            y.append(scaled_data[i + sequence_length])
        
        return np.array(X), np.array(y), scaler

    def train_model(self, df, target_column, model_name, model_type='random_forest'):
        """
        Entrena el modelo de predicción
        """
        X, y, scaler = self.prepare_data(df, target_column)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Reshape para modelos que no son de secuencia
        if model_type in ['linear_regression', 'svr']:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)
        
        # Seleccionar y entrenar modelo
        model = self.available_models[model_type]
        model.fit(X_train, y_train)
        
        # Evaluar modelo
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Guardar modelo y scaler
        self.models[model_name] = model
        self.scalers[model_name] = scaler
        
        joblib.dump(model, os.path.join(self.model_path, f'{model_name}_model.joblib'))
        joblib.dump(scaler, os.path.join(self.model_path, f'{model_name}_scaler.joblib'))
        
        return {
            'mse': mse,
            'r2': r2,
            'model_type': model_type
        }

    def predict(self, df, target_column, model_name, steps_ahead=24):
        """
        Realiza predicciones usando el modelo entrenado
        """
        if model_name not in self.models:
            # Cargar modelo si no está en memoria
            model_path = os.path.join(self.model_path, f'{model_name}_model.joblib')
            scaler_path = os.path.join(self.model_path, f'{model_name}_scaler.joblib')
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.models[model_name] = joblib.load(model_path)
                self.scalers[model_name] = joblib.load(scaler_path)
            else:
                raise ValueError(f"Modelo {model_name} no encontrado")

        # Preparar datos
        scaler = self.scalers[model_name]
        scaled_data = scaler.transform(df[[target_column]])
        
        # Realizar predicciones
        predictions = []
        current_sequence = scaled_data[-24:].reshape(1, -1)
        
        for _ in range(steps_ahead):
            pred = self.models[model_name].predict(current_sequence)
            predictions.append(pred[0])
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[0, -1] = pred[0]
        
        # Invertir normalización
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        # Crear DataFrame con fechas y predicciones
        last_date = df.index[-1]
        future_dates = [last_date + timedelta(hours=i+1) for i in range(steps_ahead)]
        
        predictions_df = pd.DataFrame({
            'fecha': future_dates,
            f'prediccion_{target_column}': predictions.flatten()
        })
        
        return predictions_df

    def evaluate_model(self, df, target_column, model_name):
        """
        Evalúa el rendimiento del modelo
        """
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} no encontrado")
        
        X, y, _ = self.prepare_data(df, target_column)
        X = X.reshape(X.shape[0], -1)
        
        predictions = self.models[model_name].predict(X)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, predictions)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }

    def get_model_info(self, model_name):
        """
        Obtiene información sobre el modelo
        """
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} no encontrado")
        
        model = self.models[model_name]
        model_type = type(model).__name__
        
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        else:
            feature_importance = None
        
        return {
            'model_type': model_type,
            'feature_importance': feature_importance.tolist() if feature_importance is not None else None
        }

    def compare_models(self, df, target_column, steps_ahead=24):
        """
        Compara diferentes modelos y retorna sus métricas
        """
        results = {}
        
        for model_type in self.available_models.keys():
            model_name = f"{target_column}_{model_type}"
            try:
                # Entrenar modelo
                metrics = self.train_model(df, target_column, model_name, model_type)
                
                # Realizar predicciones
                predictions = self.predict(df, target_column, model_name, steps_ahead)
                
                results[model_type] = {
                    'metrics': metrics,
                    'predictions': predictions.to_dict(orient='records')
                }
            except Exception as e:
                results[model_type] = {
                    'error': str(e)
                }
        
        return results 
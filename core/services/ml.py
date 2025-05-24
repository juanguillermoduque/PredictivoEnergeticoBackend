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

    def train_model(self, df, model_name, model_type='random_forest'):
        """
        Entrena el modelo de predicción
        """
        
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

    def predict(self, df, model_name):
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
        
        # Realizar predicciones
        predictions = []
        
        # Invertir normalización
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        return predictions

    def evaluate_model(self, df, model_name):
        """
        Evalúa el rendimiento del modelo
        """
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} no encontrado")
        
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

    def compare_models(self, df):
        """
        Compara diferentes modelos y retorna sus métricas
        """
        results = {}
        
        for model_type in self.available_models.keys():
            model_name = f""
            try:
                # Entrenar modelo
                metrics = self.train_model(df, model_name, model_type)
                
                # Realizar predicciones
                predictions = self.predict(df, model_name)
                
                results[model_type] = {
                    'metrics': metrics,
                    'predictions': predictions.to_dict(orient='records')
                }
            except Exception as e:
                results[model_type] = {
                    'error': str(e)
                }
        
        return results 
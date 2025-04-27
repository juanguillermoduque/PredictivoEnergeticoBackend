from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class PredictionService:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = None
        self.sequence_length = 24

    def prepare_data(self, df):
        data = self.scaler.fit_transform(df['demanda_real'].values.reshape(-1, 1))
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
            
        return np.array(X), np.array(y)

    def train_model(self, X, y):
        self.model = Sequential([
            LSTM(50, activation='relu', input_shape=(self.sequence_length, 1)),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        return self.model.fit(X, y, epochs=50, validation_split=0.2)

    def predict(self, X):
        predictions = self.model.predict(X)
        return self.scaler.inverse_transform(predictions)
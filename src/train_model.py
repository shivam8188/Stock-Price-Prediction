import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load data
data = pd.read_csv(os.path.join(BASE_DIR, "data", "stock_data.csv"), header=[0, 1])
data.columns = data.columns.get_level_values(0)
data = data[pd.to_numeric(data['Close'], errors='coerce').notna()]
data = data['Close'].astype(float).values.reshape(-1, 1)

# Scale
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Build sequences
X, y = [], []
for i in range(60, len(data_scaled)):
    X.append(data_scaled[i-60:i])
    y.append(data_scaled[i])
X = np.array(X)
y = np.array(y)

# Build model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X, y, epochs=20, batch_size=32)

# Save weights only (version-independent)
weights_path = os.path.join(BASE_DIR, "models", "lstm_weights.weights.h5")
model.save_weights(weights_path)
print("Weights saved successfully!")
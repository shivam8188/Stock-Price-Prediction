import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

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

# Build model using Input() layer instead of input_shape
inputs = keras.Input(shape=(60, 1))
x = layers.LSTM(50, return_sequences=True)(inputs)
x = layers.Dropout(0.2)(x)
x = layers.LSTM(50)(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1)(x)

model = keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X, y, epochs=20, batch_size=32)

# Save in new format
model.save(os.path.join(BASE_DIR, "models", "lstm_model.keras"))
print("Model saved successfully!")
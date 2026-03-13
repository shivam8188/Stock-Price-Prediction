import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

data = pd.read_csv("data/stock_data.csv")

data = data['Close'].values.reshape(-1,1)

scaler = MinMaxScaler()

data_scaled = scaler.fit_transform(data)

X = []
y = []

for i in range(60, len(data_scaled)):
    
    X.append(data_scaled[i-60:i])
    y.append(data_scaled[i])

X = np.array(X)
y = np.array(y)

model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(50))
model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

model.fit(X,y,epochs=20,batch_size=32)

model.save("models/lstm_model.h5")
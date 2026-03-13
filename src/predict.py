import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

model = load_model("models/lstm_model.h5")

data = pd.read_csv("data/stock_data.csv")

data = data['Close'].values.reshape(-1,1)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

X_test = []

for i in range(60,len(data_scaled)):
    
    X_test.append(data_scaled[i-60:i])

X_test = np.array(X_test)

predictions = model.predict(X_test)

predictions = scaler.inverse_transform(predictions)

print(predictions[:5])
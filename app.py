import os
import warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

st.title("Stock Price Prediction System")

stock = st.text_input("Enter Stock Symbol", "AAPL")

data = yf.download(stock, start="2015-01-01")

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

st.subheader("Stock Data (In USD)")
st.write(data.tail())

st.subheader("Closing Price Chart")
st.line_chart(data['Close'])

# ── Future Prediction ──────────────────────────────────────────────────────────
st.subheader("Future Price Prediction")

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    close = data['Close'].dropna().values
    dates = data['Close'].dropna().index

    # Scale
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close.reshape(-1, 1))

    # Rebuild same model architecture
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")

    # Load weights
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(BASE_DIR, "models", "lstm_weights.weights.h5")
    model.load_weights(weights_path)

    # Predict next 30 days
    sequence = list(scaled[-60:].flatten())
    future_preds = []

    for _ in range(30):
        X = np.array(sequence[-60:]).reshape(1, 60, 1)
        pred = model.predict(X, verbose=0)[0][0]
        future_preds.append(pred)
        sequence.append(pred)

    future_preds = scaler.inverse_transform(
        np.array(future_preds).reshape(-1, 1)
    ).flatten()

    future_dates = pd.bdate_range(
        start=dates[-1] + timedelta(days=1), periods=30
    )

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, close, label="Historical", color="blue")
    ax.plot(future_dates, future_preds, label="Predicted (30 days)", color="red", linestyle="--")
    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    st.pyplot(fig)

    # Table
    st.subheader("Predicted Prices (Next 30 Days)")
    pred_df = pd.DataFrame({
        "Date": future_dates.strftime("%d %b %Y"),
        "Predicted Price (USD)": [f"${p:.2f}" for p in future_preds]
    })
    st.dataframe(pred_df, hide_index=True)

except Exception as e:
    st.error(f"Prediction failed: {e}")
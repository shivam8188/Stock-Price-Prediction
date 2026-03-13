import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.title("Stock Price Prediction System")

stock = st.text_input("Enter Stock Symbol", "AAPL")

data = yf.download(stock,start="2015-01-01")

st.subheader("Stock Data (In USD)")

st.write(data.tail())

st.subheader("Closing Price Chart")

st.line_chart(data['Close'])
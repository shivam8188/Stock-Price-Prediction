import yfinance as yf
import pandas as pd

def load_data(stock="AAPL"):
    
    data = yf.download(stock, start="2015-01-01", end="2024-01-01")

    data.to_csv("data/stock_data.csv")

    return data

if __name__ == "__main__":
    load_data()
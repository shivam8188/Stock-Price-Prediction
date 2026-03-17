# import yfinance as yf
# import pandas as pd

# def load_data(stock="AAPL"):
    
#     data = yf.download(stock, start="2015-01-01", end="2024-01-01")

#     data.to_csv("data/stock_data.csv")

#     return data

# if __name__ == "__main__":
#     load_data()



import yfinance as yf
import pandas as pd
import os

def load_data(stock="AAPL"):
    data = yf.download(stock, start="2015-01-01")  # fresh up to today
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data.to_csv(os.path.join(BASE_DIR, "data", "stock_data.csv"))
    
    return data

if __name__ == "__main__":
    load_data()
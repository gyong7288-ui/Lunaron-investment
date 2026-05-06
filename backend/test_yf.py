import yfinance as yf
import pandas as pd

def test_data():
    ticker = "VOO"
    print(f"Testing ticker: {ticker}")
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5d")
        if hist.empty:
            print("FAILED: Dataframe is empty.")
        else:
            print("SUCCESS: Data received!")
            print(hist.tail())
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_data()

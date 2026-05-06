from yahooquery import Ticker
import pandas as pd

def test_yahooquery():
    ticker_symbol = "VOO"
    print(f"Testing with yahooquery: {ticker_symbol}")
    try:
        t = Ticker(ticker_symbol)
        hist = t.history(period="5d", interval="1d")
        if isinstance(hist, pd.DataFrame) and not hist.empty:
            print("SUCCESS: Data received via yahooquery!")
            print(hist.tail())
        else:
            print(f"FAILED: Result is not as expected. {type(hist)}")
            print(hist)
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_yahooquery()

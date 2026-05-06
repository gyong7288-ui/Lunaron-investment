import yfinance as yf
import requests

def test_data():
    ticker = "VOO"
    print(f"Testing ticker with session: {ticker}")
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })
    try:
        t = yf.Ticker(ticker, session=session)
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

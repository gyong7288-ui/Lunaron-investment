from yahooquery import Ticker
import pandas as pd

def test_news():
    t = Ticker("VOO")
    print("--- News ---")
    news = t.news(5)
    print(f"News Type: {type(news)}")
    if isinstance(news, list) and len(news) > 0:
        print(f"Item 0 Type: {type(news[0])}")
        print(f"Item 0 Content: {news[0]}")
    
    # Try another way for news if needed
    # t.search() often contains news

    print("\n--- Summary ---")
    summary = t.summary_detail.get("VOO", {})
    print(f"Summary: {summary.get('longBusinessSummary', 'No summary')}")

if __name__ == "__main__":
    test_news()

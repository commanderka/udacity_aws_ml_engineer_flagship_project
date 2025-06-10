import pickle
from news_fetcher import fetch_news

if __name__ == "__main_":
    ticker_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA","NESR","NIO","NVDA","META","NFLX", "ALV","CHV","MBG"]
    news_sentiment_dict = fetch_news(ticker_symbols)  
    with open("news_sentiment.pkl", "wb") as f:
        pickle.dump(news_sentiment_dict, f)
    print("News fetched successfully.")

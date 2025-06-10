from typing import List
from polygon import RESTClient
from datetime import datetime
from time import sleep
import pickle

def fetch_news(ticker_symbols:List[str]):
    
    client = RESTClient("RtGfOcVRoi9Phqlk_23p6vsokRbuDDbL")

    # Display the title and insights for each article
    sentiment_dict = {}
    start_date = "2025-03-01"  # Reset start date for each ticker
    last_updated_date = ""
    for ticker_symbol in ticker_symbols:
        print("Processing ticker symbol:", ticker_symbol, flush=True)
        ticker_dict = {}
        sentiment_dict[ticker_symbol] = ticker_dict
        end_date = datetime.now().strftime("%Y-%m-%d")  # Use current date as end date
        completed =False
        while (not completed):
            try:
                # Fetch news articles with insights
                news_articles = client.list_ticker_news(
                    ticker_symbol, 
                    params={"published_utc.gte": start_date, "published_utc.lte": end_date}, 
                    order="desc", 
                    )
                for article in news_articles:
                    #print(f"{article.title} [Insights: {article.insights}]")
                    #format utc string as date string
                    date_string = datetime.strptime(article.published_utc, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d")
                    if date_string not in ticker_dict:
                        ticker_dict[date_string] = []
                    for insight in article.insights:
                        ticker_dict[date_string].append(insight.sentiment)
                    last_updated_date = datetime.strptime(article.published_utc, "%Y-%m-%dT%H:%M:%SZ")
                    end_date = last_updated_date.strftime("%Y-%m-%d")   
                completed = True     
            except Exception as e:
                print(f"Error fetching news articles: {e}", flush=True)
                print("Refetching with end date:", end_date, flush=True)
                print("Dumping intermediate results to avoid data loss", flush=True)
                #dump intermediate results to avoid data loss
                with open("news_sentiment_intermediate.pkl", "wb") as f:
                    pickle.dump(sentiment_dict, f)
                #sleep for 30 seconds to avoid hitting the API rate limit
                sleep(30)
    #dump new dict with pickle
    return sentiment_dict

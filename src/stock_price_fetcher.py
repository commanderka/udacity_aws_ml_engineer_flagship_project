import yfinance as yf
import pandas as pd
import pickle
from datetime import datetime
from news_fetcher import fetch_news
import os
import numpy as np

def get_historic_stock_prices(ticker_symbols, period='20y'):
    ticker_symbol_str = " ".join(ticker_symbols)
    historic_data = yf.download(ticker_symbol_str, period=period)
    flat_df = historic_data.stack(level=1).reset_index()
    #compute the average price for each ticker symbol
    flat_df['Average'] = (flat_df['Open'] + flat_df['Close']) / 2
    return flat_df

#loads the sentiment data and join it with the stock prices using the time period of the sentiment data
def get_stock_prices_with_sentiments(ticker_symbols):
   
    #load the sentiment data
    #get dir of current file

    
    current_file_path = os.path.abspath(__file__)
    current_file_dir = os.path.dirname(current_file_path)
    sentiment_file_path = os.path.join(current_file_dir, "news_sentiment.pkl")
    if not os.path.exists(sentiment_file_path):
        print("Sentiment file not found. Please run the news fetcher first.")
        return None
    with open(sentiment_file_path, "rb") as f:
        news_sentiment_dict = pickle.load(f)
    
    #get the stock prices
    flat_df = get_historic_stock_prices(ticker_symbols, period='1y')

    #remove ticker symbols for which we don't have sentiment data
    ticker_symbols = [ticker for ticker in ticker_symbols if ticker in news_sentiment_dict and news_sentiment_dict[ticker]]
    if not ticker_symbols:
        print("No ticker symbols found with sentiment data.")
        return None
    #filter the flat_df to only include the ticker symbols we have sentiment data for
    flat_df = flat_df[flat_df['Ticker'].isin(ticker_symbols)]
    
    #convert the date column to datetime
    flat_df['Date'] = pd.to_datetime(flat_df['Date'])
    
    #join the sentiment data with the stock prices using the date column
    #add sentiment column to the flat_df
    flat_df['Sentiment'] = np.nan # Initialize sentiment column with default value (0.5 is neutral sentiment)
    for ticker_symbol in ticker_symbols:
        if ticker_symbol in news_sentiment_dict:
            ticker_sentiments = news_sentiment_dict[ticker_symbol]
            for date_str, sentiments in ticker_sentiments.items():
                date = datetime.strptime(date_str, "%Y-%m-%d")
                #count the number of positive sentiments
                n_positives = sum(1 for sentiment in sentiments if sentiment == "positive")
                avg_sentiment = n_positives / len(sentiments) if sentiments else np.nan
                #set sentiment to avg_sentiment if ticker symbol and date exist in the flat_df
                #add sentiment col to flat_df and set value to avg_sentiment if ticker symbol and date match        
                flat_df.loc[(flat_df["Ticker"] == ticker_symbol) & (flat_df["Date"] == date),"Sentiment"] = avg_sentiment
    
    #restrict flat_df dates to the date range of the sentiment data
    min_date = min([datetime.strptime(date_str, "%Y-%m-%d") for date_str in news_sentiment_dict[ticker_symbols[0]].keys()])
    max_date = max([datetime.strptime(date_str, "%Y-%m-%d") for date_str in news_sentiment_dict[ticker_symbols[0]].keys()])
    flat_df = flat_df[(flat_df['Date'] >= min_date) & (flat_df['Date'] <= max_date)]
    #set nan sentiment values to the previous sentiment value for the same ticker symbol
    flat_df['Sentiment'] = flat_df.groupby('Ticker')['Sentiment'].ffill().bfill()
    return flat_df

def fill_missing_dates_and_add_additional_cols(df,ticker_symbol):
    flat_df_ticker = df[df['Ticker'] == ticker_symbol]
    flat_df_ticker.drop(columns=['Ticker'], inplace=True)
    #add missing dates and set value to the previous column value
    #add another column is_business day and set it to False fpr the newly added rows
    flat_df_ticker['Date'] = pd.to_datetime(flat_df_ticker['Date'])
    flat_df_ticker['is_business_day'] = True
    #set the date column as index and resample to daily frequency, filling missing dates with the previous value
    flat_df_ticker = flat_df_ticker.set_index('Date')
    all_dates = pd.date_range(flat_df_ticker.index.min(), flat_df_ticker.index.max(), freq='D')
    missing_dates = all_dates.difference(flat_df_ticker.index)
    flat_df_ticker = flat_df_ticker.asfreq('D').fillna(method='ffill').reset_index()
    flat_df_ticker.loc[flat_df_ticker["Date"].isin(missing_dates),"is_business_day"] = False

    return flat_df_ticker


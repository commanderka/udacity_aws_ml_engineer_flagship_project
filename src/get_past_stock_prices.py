import yfinance as yf
import pandas as pd
from darts import TimeSeries
from darts.models import ExponentialSmoothing, AutoARIMA, XGBModel, NBEATSModel, RNNModel, TCNModel
import numpy as np
from darts.metrics import mape
from matplotlib import pyplot as plt
from news_fetcher import fetch_news
import pickle 
from stock_price_fetcher import get_historic_stock_prices

if __name__ == "__main__":


    ticker_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA","NESR","NIO","NVDA","META","NFLX", "ALV","CHV","MBG"]
    
    ticker_symbol_str = " ".join(ticker_symbols)
    flat_df = get_historic_stock_prices(ticker_symbols, period='20y')
    #transform the dataframe such that there is one col for the ticker symbol, a col for the open price, a col for the close price and a col for the volume

    #take log for volume
    flat_df['Volume'] = flat_df['Volume'].apply(lambda x: np.log(x) if x > 0 else 0)
    cols_of_interest = ['Average']
    #just consider one ticker symbol
    flat_df_google = flat_df[flat_df['Ticker'] == 'GOOGL']
    flat_df_google.drop(columns=['Ticker'], inplace=True)
    #add missing dates and set value to the previous column value
    #add another column is_business day and set it to False fpr the newly added rows
    flat_df_google['Date'] = pd.to_datetime(flat_df_google['Date'])
    flat_df_google['is_business_day'] = True
    #set the date column as index and resample to daily frequency, filling missing dates with the previous value
    flat_df_google = flat_df_google.set_index('Date').asfreq('D').fillna(method='ffill').reset_index()
    #value_cols = [col for col in flat_df_google.columns if col not in ['Date']]
    value_cols = ["Average"]
    past_covariates = TimeSeries.from_dataframe(flat_df_google, value_cols=["Close","High","Low","Volume"], time_col='Date', freq="D", fill_missing_dates=False)
    series = TimeSeries.from_dataframe(flat_df_google,time_col='Date',                                      
                                      value_cols=value_cols, fill_missing_dates=False)
    
    #model = XGBModel(lags=30,lags_past_covariates=30,)
    #model = NBEATSModel(input_chunk_length=30, output_chunk_length=10, n_epochs=10, random_state=42)
    #model = AutoARIMA()
    #model = TCNModel(input_chunk_length=30, output_chunk_length=10, n_epochs=10, random_state=42)
    model = ExponentialSmoothing()
    train, test = series.split_after(pd.Timestamp('2020-01-01'))
    past_covariates_train, past_covariates_test = past_covariates.split_after(pd.Timestamp('2020-01-01'))
    #model.fit(train, past_covariates=past_covariates_train)
    model.fit(train)
    #forecast = model.predict(len(test), past_covariates=past_covariates)
    forecast = model.predict(len(test))
    series.plot()
    forecast.plot(label="forecast")
    plt.savefig("forecast.png")
    print(f"Mean absolute percentage error: {mape(series, forecast):.2f}%.")
    pass
import argparse
import boto3
import pickle
from stock_price_fetcher import get_historic_stock_prices
from darts import TimeSeries
import pandas as pd
import numpy as np
from darts.models import Prophet, XGBModel
from darts.metrics import mape


def train_and_evaluate(args):
    s3 = boto3.resource('s3')
    bucket_name = "stock-market-data-udacity-flagship"
    news_sentiments = pickle.loads(s3.Bucket(bucket_name).Object("key_to_pickle.pickle").get()['Body'].read())

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
    
    model = XGBModel(lags=args.lag,lags_past_covariates=args.lag,)
    #model = NBEATSModel(input_chunk_length=30, output_chunk_length=10, n_epochs=10, random_state=42)
    #model = AutoARIMA()
    #model = TCNModel(input_chunk_length=30, output_chunk_length=10, n_epochs=10, random_state=42)
    model = Prophet()
    train, test = series.split_after(pd.Timestamp('2020-01-01'))
    past_covariates_train, past_covariates_test = past_covariates.split_after(pd.Timestamp('2020-01-01'))
    model.fit(train, past_covariates=past_covariates_train)
    #model.fit(train)
    model.fit()
    forecast = model.predict(len(test), past_covariates=past_covariates)
    #forecast = model.predict(len(test))
    print(f"Mean absolute percentage error: {mape(series, forecast):.2f}%.")



def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )
    args = parser.parse_args()

    

    


if __name__ == "__main__":
    main()

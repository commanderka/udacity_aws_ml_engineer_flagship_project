import argparse
from stock_price_fetcher import get_historic_stock_prices
from darts import TimeSeries
import pandas as pd
import numpy as np
from darts.models import Prophet, XGBModel, NBEATSModel
from darts.metrics import mape, rmse, r2_score
import os


def train_and_evaluate(args):

    ticker_symbols = ["AAPL", "GOOGL"]
    
    flat_df = get_historic_stock_prices(ticker_symbols, period='20y')
    #transform the dataframe such that there is one col for the ticker symbol, a col for the open price, a col for the close price and a col for the volume

    #take log for volume
    flat_df['Volume'] = flat_df['Volume'].apply(lambda x: np.log(x) if x > 0 else 0)
    mape_metric_list = []

    #save flat df to be able to load the past covariates for the ticker symbol later
    flat_df.to_csv(os.path.join(args.model_dir, "flat_df.csv"), index=False)

    for ticker_symbol in ticker_symbols:
        flat_df_ticker = flat_df[flat_df['Ticker'] == ticker_symbol]
        flat_df_ticker.drop(columns=['Ticker'], inplace=True)
        #add missing dates and set value to the previous column value
        #add another column is_business day and set it to False for the newly added rows
        flat_df_ticker['Date'] = pd.to_datetime(flat_df_ticker['Date'])
        flat_df_ticker['is_business_day'] = True
        #set the date column as index and resample to daily frequency, filling missing dates with the previous value
        flat_df_ticker = flat_df_ticker.set_index('Date').asfreq('D').fillna(method='ffill').reset_index()
        #value_cols = [col for col in flat_df_google.columns if col not in ['Date']]
        value_cols = ["Average"]
        past_covariates = TimeSeries.from_dataframe(flat_df_ticker, value_cols=["Close","High","Low","Volume"], time_col='Date', freq="D", fill_missing_dates=False)
        series = TimeSeries.from_dataframe(flat_df_ticker,time_col='Date',                                      
                                        value_cols=value_cols, fill_missing_dates=False)
        model = NBEATSModel(input_chunk_length=args.lag, output_chunk_length=10, n_epochs=args.epochs, random_state=42)
        train, test = series.split_after(0.8)
        past_covariates_train, _ = past_covariates.split_after(0.8)

        model.fit(train, past_covariates=past_covariates_train, )
        model.save(os.path.join(args.model_dir, f"{ticker_symbol}_model.pth"))
        forecast = model.predict(len(test), past_covariates=past_covariates)
          
        mape_metric = mape(series, forecast) 
        mape_metric_list.append(mape_metric)  
      

    #compute median mape metric
    median_mape = np.median(mape_metric_list)
    print(f"Median Mean Absolute Percentage Error (MAPE): {median_mape:.2f}%.")

            

def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--lag",
        type=int,
        default=10,
        metavar="N",
        help="Lag for time series prediction (default: 10)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="Number of epochs to train (default: 10)",
    )
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])

    args = parser.parse_args()
    train_and_evaluate(args)

    

    
if __name__ == "__main__":
    main()

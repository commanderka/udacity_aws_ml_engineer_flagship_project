import yfinance as yf
import pandas as pd
from darts import TimeSeries
from darts.models import Prophet, NBEATSModel, TiDEModel
import numpy as np
from darts.metrics import mape, r2_score, rmse
from matplotlib import pyplot as plt
from news_fetcher import fetch_news
import pickle 
from stock_price_fetcher import get_historic_stock_prices
from collections import defaultdict
import warnings
import os

if __name__ == "__main__":

    current_file_dir = os.path.dirname(current_file_path = os.path.abspath(__file__)) 
    ticker_symbols = ["AAPL", "GOOGL"]
    
    ticker_symbol_str = " ".join(ticker_symbols)
    flat_df = get_historic_stock_prices(ticker_symbols, period='20y')
    #transform the dataframe such that there is one col for the ticker symbol, a col for the open price, a col for the close price and a col for the volume

    #take log for volume
    flat_df['Volume'] = flat_df['Volume'].apply(lambda x: np.log(x) if x > 0 else 0)
    cols_of_interest = ['Average']
    metrics_for_models = defaultdict(lambda: defaultdict(list))

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
        models = [NBEATSModel(input_chunk_length=30, output_chunk_length=10, n_epochs=2, random_state=42),TiDEModel(input_chunk_length=30, output_chunk_length=10, n_epochs=2, random_state=42),Prophet()]
        #models = [XGBModel(lags=30,lags_past_covariates=30,), AutoARIMA(),ExponentialSmoothing()]
        uses_past_covariates_list = [True, True, False]
        train, test = series.split_after(0.8)
        past_covariates_train, past_covariates_test = past_covariates.split_after(0.8)
        series.plot()
        for model_idx, model in enumerate(models):
            uses_past_covariates = uses_past_covariates_list[model_idx]

            if uses_past_covariates:
                model.fit(train, past_covariates=past_covariates_train, )
                forecast = model.predict(len(test), past_covariates=past_covariates)
            else:
                model.fit(train)
                forecast = model.predict(len(test))   
            forecast.plot(label="forecast_" + model.__class__.__name__)

            mape_metric = mape(series, forecast)   
            rmse_metric = rmse(series, forecast)
            r2_metric = r2_score(series, forecast)
            metrics_for_models["mape"][model.__class__.__name__].append(mape_metric)
            metrics_for_models["rmse"][model.__class__.__name__].append(rmse_metric)
            metrics_for_models["r2"][model.__class__.__name__].append(r2_metric)
        plt.savefig(os.path.join(current_file_dir,f"forecast_sophisticated_{ticker_symbol}.png"))
        plt.close()
            
    #compute average for each model
    for metric, model_metrics in metrics_for_models.items():
        print(f"Average {metric} for each model:")
        for model_name, values in model_metrics.items():
            avg_value = np.mean(values)
            print(f"{model_name}: {avg_value:.4f}")
        print("\n")
    #also print the median
    for metric, model_metrics in metrics_for_models.items():
        print(f"Median {metric} for each model:")
        for model_name, values in model_metrics.items():
            median_value = np.median(values)
            print(f"{model_name}: {median_value:.4f}")
        print("\n")

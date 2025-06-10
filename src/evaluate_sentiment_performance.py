from stock_price_fetcher import get_stock_prices_with_sentiments, fill_missing_dates_and_add_additional_cols
from darts import TimeSeries
from darts.models import ExponentialSmoothing, AutoARIMA, XGBModel, NBEATSModel, RNNModel, TCNModel, CatBoostModel, RegressionModel
from sklearn.linear_model import Ridge
from darts.models.forecasting.tide_model import TiDEModel
from darts.models.forecasting.random_forest import RandomForest
import pandas as pd
from matplotlib import pyplot as plt
from darts.metrics import mape,r2_score,rmse
import numpy as np

if __name__ == "__main__":
    stock_prices_with_sentiments = get_stock_prices_with_sentiments(["GOOGL", "AAPL", "MSFT", "AMZN", "TSLA","NESR","NIO","NVDA","META","NFLX", "ALV","CHV","MBG"])
    #evalute the prediction performance with and without sentiments
    n_lags = 10
    mape_dict = {
        "with_sentiment": [[] for _ in range(1,n_lags+1)],
        "without_sentiment": [[] for _ in range(1,n_lags+1)]
    }
    r2_score_dict = {
        "with_sentiment": [[] for _ in range(1,n_lags+1)],
        "without_sentiment": [[] for _ in range(1,n_lags+1)]
    }
    rmse_dict = {
        "with_sentiment": [[] for _ in range(1,n_lags+1)],
        "without_sentiment": [[] for _ in range(1,n_lags+1)]
    }

    for lag in range(1,n_lags+1):
        for ticker_symbol in stock_prices_with_sentiments["Ticker"].unique():
            for mode in ["with_sentiment", "without_sentiment"]:
                stock_prices_with_sentiments_and_additional_features_ticker = fill_missing_dates_and_add_additional_cols(stock_prices_with_sentiments, ticker_symbol)
                stock_prices_with_sentiments_and_additional_features_ticker['Volume'] = stock_prices_with_sentiments_and_additional_features_ticker['Volume'].apply(lambda x: np.log(x) if x > 0 else 0)

                value_cols = ["Average"]
                if mode == "with_sentiment":
                    past_covariates = TimeSeries.from_dataframe(stock_prices_with_sentiments_and_additional_features_ticker, value_cols=["Close","High","Low","Volume","is_business_day","Sentiment"], time_col='Date', freq="D", fill_missing_dates=False)
                else:
                    past_covariates = TimeSeries.from_dataframe(stock_prices_with_sentiments_and_additional_features_ticker, value_cols=["Close","High","Low","Volume","is_business_day"], time_col='Date', freq="D", fill_missing_dates=False)
                series = TimeSeries.from_dataframe(stock_prices_with_sentiments_and_additional_features_ticker,time_col='Date',value_cols=value_cols, fill_missing_dates=False)
                
                model = RandomForest(lags=lag,lags_past_covariates=lag,n_estimators=10, max_depth=5, random_state=42)
                #model = NBEATSModel(input_chunk_length=30, output_chunk_length=10, n_epochs=10, random_state=42)
                #model = AutoARIMA()
                #model = TCNModel(input_chunk_length=30, output_chunk_length=10, n_epochs=10, random_state=42)
                train, test = series.split_after(pd.Timestamp('2025-05-15'))
                past_covariates_train, past_covariates_test = past_covariates.split_after(pd.Timestamp('2025-05-15'))
                model.fit(train, past_covariates=past_covariates_train)
                forecast = model.predict(len(test), past_covariates=past_covariates)
                mape_dict[mode][lag-1].append(mape(series, forecast))
                r2_score_dict[mode][lag-1].append(r2_score(series, forecast))
                rmse_dict[mode][lag-1].append(rmse(series, forecast))
        

     #compute the mean of the metrics for each lag
    mape_mean = {key: [sum(values)/len(values) for values in mape_dict[key]] for key in mape_dict}
    r2_score_mean = {key: [sum(values)/len(values) for values in r2_score_dict[key]] for key in r2_score_dict}
    rmse_mean = {key: [sum(values)/len(values) for values in rmse_dict[key]] for key in rmse_dict}
    
    #plot the metrics for with and without sentiment
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 11), mape_mean["with_sentiment"], label="With Sentiment", marker='o')
    plt.plot(range(1, 11), mape_mean["without_sentiment"], label="Without Sentiment", marker='o')
    plt.title("MAPE for Stock Price Prediction with and without Sentiment")
    plt.xlabel("Lag")
    plt.ylabel("Mean Absolute Percentage Error (MAPE)")
    plt.legend()
    plt.grid()
    plt.savefig("mape_with_without_sentiment.png")
    plt.show()
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 11), r2_score_mean["with_sentiment"], label="With Sentiment", marker='o')
    plt.plot(range(1, 11), r2_score_mean["without_sentiment"], label="Without Sentiment", marker='o')
    plt.title("R2 Score for Stock Price Prediction with and without Sentiment")
    plt.xlabel("Lag")
    plt.ylabel("R2 Score")
    plt.legend()
    plt.grid()
    plt.savefig("r2_score_with_without_sentiment.png")
    plt.show()
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 11), rmse_mean["with_sentiment"], label="With Sentiment", marker='o')
    plt.plot(range(1, 11), rmse_mean["without_sentiment"], label="Without Sentiment", marker='o')
    plt.title("RMSE for Stock Price Prediction with and without Sentiment")
    plt.xlabel("Lag")
    plt.ylabel("Root Mean Squared Error (RMSE)")
    plt.legend()
    plt.grid()
    plt.savefig("rmse_with_without_sentiment.png")
    plt.show()
    #save the metrics to a csv file
            

    pass



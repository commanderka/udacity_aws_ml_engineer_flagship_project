import logging
import os
import sys
import pandas as pd
from darts import TimeSeries
from collections import defaultdict

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

from fastapi import FastAPI, Request
from darts.models import NBEATSModel

 
app = FastAPI()

print("Loading model...")
model_dir = "/opt/ml/model"
#load models for all ticker symbols

#create a default dict of default dicts

app.model_dict = defaultdict(lambda: defaultdict(dict))
ticker_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA","NESR","NIO","NVDA","META","NFLX", "ALV","CHV","MBG"]
#load the flat_df for the ticker symbols
flat_df_path = os.path.join(model_dir, "flat_df.csv")
if os.path.exists(flat_df_path):
    flat_df = pd.read_csv(flat_df_path)
    #create a TimeSeries object for each ticker symbol
    for ticker_symbol in ticker_symbols:
        flat_df_ticker = flat_df[flat_df['Ticker'] == ticker_symbol]
        flat_df_ticker.drop(columns=['Ticker'], inplace=True)
        #add missing dates and set value to the previous column value
        #add another column is_business day and set it to False for the newly added rows
        flat_df_ticker['Date'] = pd.to_datetime(flat_df_ticker['Date'])
        flat_df_ticker['is_business_day'] = True
        #set the date column as index and resample to daily frequency, filling missing dates with the previous value
        flat_df_ticker = flat_df_ticker.set_index('Date').asfreq('D').fillna(method='ffill').reset_index()
        past_covariates = TimeSeries.from_dataframe(flat_df_ticker, value_cols=["Close","High","Low","Volume"], time_col='Date', freq="D", fill_missing_dates=False)

        model_path = os.path.join(model_dir, f"{ticker_symbol}_model.pth")
        if os.path.exists(model_path):
            # Load the model and the covariates
            app.model_dict[ticker_symbol]["model"] = NBEATSModel.load(model_path)
            app.model_dict[ticker_symbol]['past_covariates'] =  past_covariates
            logger.info(f"Loaded model for {ticker_symbol} from {model_path}")
        else:
            logger.warning(f"Model file for {ticker_symbol} not found at {model_path}")
        
    logger.info(f"Loaded flat_df from {flat_df_path}")
else:
    logger.error(f"Flat_df file not found at {flat_df_path}")
    raise FileNotFoundError(f"Flat_df file not found at {flat_df_path}")

 
@app.get('/ping')
async def ping():
    return {"message": "ok"}
 

@app.post('/invocations')
async def invocations(request: Request):
    # Assuming input_data is a dictionary with 'ticker_symbol' and 'n_days'
    input_data = await request.json()
    predictions = []
    for instance in input_data["instances"]:
        print(f"Input data:{input_data}")
        ticker_symbol = instance['ticker_symbol']
        current_model_dict = app.model_dict[ticker_symbol]
        past_covariates = current_model_dict['past_covariates']
        current_model = current_model_dict['model']
        n_days = instance['n_days']
        prediction = current_model.predict(n=int(n_days), past_covariates=past_covariates)
        #convert timeseries to list
        prediction_for_instance = prediction.to_dataframe().reset_index().to_dict(orient="records")
        predictions.append(prediction_for_instance)
    return predictions

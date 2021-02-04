from os import write
import pandas as pd
from pandas_datareader import data as pdr
import datetime

def get_stock_data(tag: str, start_date: datetime.datetime, end_date: datetime.datetime) -> pd.core.frame.DataFrame:
    if type(tag) != type([]):
        tag = [tag]
    try:
        return pdr.get_data_yahoo(tag, start=start_date, end=end_date)
    except :
        return pd.DataFrame.empty()

def get_predictive_model(tag, start_date = pd.to_datetime('2017-01-01'), end_date = datetime.datetime.today()):
    data = get_stock_data(tag, start_date, end_date)
    print(type(data))
    
if __name__ == '__main__':
    get_predictive_model('gme')
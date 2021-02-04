from os import write
from typing import ClassVar
import pandas as pd
from pandas_datareader import data as pdr
import datetime
import matplotlib.pyplot as plt

def get_stock_data(tag: str, start_date: datetime.datetime, end_date: datetime.datetime) -> pd.core.frame.DataFrame:
    if type(tag) != type([]):
        tag = [tag]
    try:
        data = pdr.get_data_yahoo(tag, start=start_date, end=end_date)
        data.reset_index(inplace=True,drop=False)
        
        data['Mid'] = (data['Low']+data['High'])/2.0
        return data
    except :
        print('brrrrr')
        return pd.DataFrame()


def real_data_graph(df: pd.core.frame.DataFrame, tag:str, xticks = 5) -> None:
    xticks = int(df.shape[0] / xticks)
    df['date_string'] = df['Date'].dt.strftime('%Y-%m-%d')  
    plt.figure(figsize = (18,9))
    plt.plot(range(df.shape[0]), df['Mid'])
    plt.xticks(range(0,df.shape[0],xticks),df['date_string'].loc[::xticks],rotation=45)
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Mid Price',fontsize=18)
    plt.savefig('img/current.png')
    plt.savefig(f'img/{tag}.png')
    

def get_predictive_model(tag:str, start_date = pd.to_datetime('2017-01-01'), end_date = datetime.datetime.today()):
    df = get_stock_data(tag, start_date, end_date)

    if df.empty:
        return

    df.to_csv(f'data/{tag}.csv')
    real_data_graph(df, tag)



    
    
if __name__ == '__main__':
    get_predictive_model('gme', pd.to_datetime('2021-1-1'))
from os import write
import pandas as pd
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf 
import datetime
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler

rcParams['figure.figsize']=20,10

def get_stock_data(tag, start_date, end_date):
    if type(tag) != type([]):
        tag = [tag]
    try:
        return pdr.get_data_yahoo(tag, start=start_date, end=end_date)
    except :
        return pd.DataFrame.empty()

def get_predictive_model(tag, start_date = pd.to_datetime('2017-01-01'), end_date = datetime.datetime.today()):
    data = get_stock_data([tag], start_date, end_date)
    print(data)
    
if __name__ == '__main__':
    get_predictive_model('gme')
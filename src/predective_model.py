from pandas_datareader import data as pdr

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler

def get_stock_data(tag: str, start_date: dt.datetime, end_date: dt.datetime) -> pd.core.frame.DataFrame:
    # forces tag into a list
    tag = [tag]
    # attempts to pull the data
    try:
        # get it from yahoo
        data = pdr.get_data_yahoo(tag, start=start_date, end=end_date)
        # reset the index to remove exta symbols
        data.reset_index(inplace=True,drop=False)
        # create a shorter data string
        data['date_string'] = data['Date'].dt.strftime('%Y-%m-%d') 
        data = data.drop(["Adj Close", "Volume"], axis = 1)


        # returns the data
        return data
    except :
        # if the data is wrong, return a blank one 
        return pd.DataFrame()


def get_predictive_model(tag:str, start_date = pd.to_datetime('2021-01-01'), end_date = dt.datetime.today()):
    df = get_stock_data(tag, start_date, end_date)
    
    high_prices = df.loc[:,'High']
    low_prices = df.loc[:,'Low']
    mid_prices = (high_prices+low_prices)/2.0

    # TODO dynamic
    train_data = mid_prices[:11000]
    test_data = mid_prices[11000:]

    # Scale the data to be between 0 and 1
    scaler = MinMaxScaler()
    train_data = train_data.reshape(-1,1)
    test_data = test_data.reshape(-1,1)

    # TODO dynamic
    smoothing_window_size = 2500
    # TODO dynamic
    for di in range(0,10000,smoothing_window_size):
        scaler.fit(train_data[di:di+smoothing_window_size,:])
        train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])

    # You normalize the last bit of remaining data
    scaler.fit(train_data[di+smoothing_window_size:,:])
    train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])

    # Reshape both train and test data
    train_data = train_data.reshape(-1)

    # Normalize test data
    test_data = scaler.transform(test_data).reshape(-1)

    # Exponential Smoothing
    EMA = 0.0
    gamma = 0.1
    
    # TODO dynamic
    for ti in range(11000):
        EMA = gamma*train_data[ti] + (1-gamma)*EMA
        train_data[ti] = EMA

    # saving here to make visualizations easier later
    all_mid_data = np.concatenate([train_data,test_data],axis=0)

    # exponential moving average
    # very accurate, much wow

    # TODO dynamic
    window_size = 100
    N = train_data.size

    run_avg_predictions = []
    run_avg_x = []

    mse_errors = []

    running_mean = 0.0
    run_avg_predictions.append(running_mean)

    decay = 0.5

    for pred_idx in range(1,N):
        if pred_idx >= N:
            date = dt.datetime.strptime(pred_idx, '%Y-%m-%d').date() + dt.timedelta(days=1)
        else:
            date = df.loc[pred_idx,'Date']

        running_mean = running_mean*decay + (1.0-decay)*train_data[pred_idx-1]
        run_avg_predictions.append(running_mean)
        mse_errors.append((run_avg_predictions[-1]-train_data[pred_idx])**2)
        run_avg_x.append(date)

    print('MSE error for EMA averaging: %.5f'%(0.5*np.mean(mse_errors)))

    plt.figure(figsize = (18,9))
    plt.plot(range(df.shape[0]),all_mid_data,color='b',label='True')
    plt.plot(range(0,N),run_avg_predictions,color='orange', label='Prediction')
    #plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Mid Price')
    plt.legend(fontsize=18)
    plt.show()

if __name__ == '__main__':
    get_predictive_model('gme')

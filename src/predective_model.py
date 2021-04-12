from pandas_datareader import data as pdr

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
import os
from platform import system as operating_system


def clear_static():
    for root, dirs, files in os.walk('src/static'):
        for file in files:
            #append the file name to the list
            print(os.path.join(root,file))
            os.remove(os.path.join(root,file))

    for root, dirs, files in os.walk('static'):
        for file in files:
            #append the file name to the list
            print(os.path.join(root,file))
            os.remove(os.path.join(root,file))

def get_stock_data(tag: str, start_date: dt.datetime, end_date: dt.datetime) -> pd.core.frame.DataFrame:
    # forces tag into a list
    tag = [tag]
    # attempts to pull the data
    try:
        # get it from yahoo
        data = pdr.get_data_yahoo(tag, start=start_date, end=end_date)
        # reset the index to remove extra symbols
        data.reset_index(inplace=True,drop=False)
        # create a shorter data string
        data['Date_String'] = data['Date'].dt.strftime('%Y-%m-%d')
        data = data.drop(["Adj Close", "Volume"], axis = 1)


        # returns the data
        return data
    except :
        # if the data is wrong, return a blank one
        print("errord")
        return pd.DataFrame()

def get_predictive_model(tag:str, start_date = pd.to_datetime('2020-01-01'), end_date = dt.datetime.today()):
    df = get_stock_data(tag, start_date, end_date)

    X = df['Close']
    # y = df['High']
    y = df['Date'].apply(lambda x:x.toordinal())



    X_train, X_test, y_train, y_test = train_test_split(X, y)

    scaler = StandardScaler()

    scaler.fit(X_train)
    StandardScaler(copy=True, with_mean=True, with_std=True)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
    mlp.fit(X_train,y_train)
    predictions = mlp.predict(X_test)
    mlp.predict()

    # plt.figure(figsize = (14,7))
    # plt.plot(range(df.shape[0]),all_mid_data,color='b',label='True')
    # plt.plot(range(0,N),run_avg_predictions,color='orange', label='Prediction')
    # plt.xticks(range(0,df.shape[0],50),df['Date_String'].loc[::50],rotation=45)
    # plt.xlabel('Date')
    # plt.ylabel('Mid Price')
    # plt.legend(fontsize=12)

    # save_loc = f'..\\img\\{tag}' if operating_system() == 'Windows' else f'../img/{tag}'

    # try:
    #     plt.savefig(save_loc)
    #     plt.savefig(f'static\\{tag}' if operating_system() == 'Windows' else f'static/{tag}')
    # except :
    #     save_loc = f'img\\{tag}' if operating_system() == 'Windows' else f'img/{tag}'
    #     plt.savefig(save_loc)
    #     plt.savefig(f'src\\static\\{tag}' if operating_system() == 'Windows' else f'src/static/{tag}')

if __name__ == '__main__':
    get_predictive_model('gme', start_date=pd.to_datetime('2020-01-01'))

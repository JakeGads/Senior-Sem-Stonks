import os
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

import datetime as dt
import pandas_datareader as pdr
import pandas as pd

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
        
        data.to_csv(f'data/{tag[0]}.csv')

        # returns the data
        return data
    except :
        # if the data is wrong, return a blank one
        
        return pd.DataFrame()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from platform import system as operating_system

def get_predictive_model(tag:str, start_date = pd.to_datetime('2020-01-01'), end_date = dt.datetime.today()):

    def create_dataset(df):
        x = []
        y = []
        for i in range(50, df.shape[0]):
            x.append(df[i-50:i, 0])
            y.append(df[i, 0])
        x = np.array(x)
        y = np.array(y)
        return x,y

    df = get_stock_data(tag, start_date, end_date)
    if df.empty:
        return False
    df = df['Close'].values

    dataset_train = np.array(df[:int(df.shape[0]*0.8)])
    dataset_test = np.array(df[int(df.shape[0]*0.8):])

    scaler = MinMaxScaler(feature_range=(0,1))
    dataset_train = scaler.fit_transform(dataset_train)
    dataset_test = scaler.transform(dataset_test)

    x_train, y_train = create_dataset(dataset_train)
    x_test, y_test = create_dataset(dataset_test)

    model = Sequential()
    model.add(LSTM(units=96, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=96, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=96, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=96))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(x_train, y_train, epochs=25, batch_size=32)
    model.save('stock_prediction.h5') # we have to save no I don't know why it just seems to strengthen the model
    model = load_model('stock_prediction.h5')

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    fig, ax = plt.subplots(figsize=(16,8))
    ax.set_facecolor('#000041')
    ax.plot(y_test_scaled, color='red', label='Original price')
    plt.plot(predictions, color='cyan', label='Predicted price')
    
    predictor = SimpleExpSmoothing(predictions).fit()

    plt.xlabel('Date Ordinal')
    plt.ylabel('Closing Price')
    plt.legend()
    

    loc = f'..\\img\\{tag}' if operating_system() == 'Windows' else f'../img/{tag}'
    try:
        plt.savefig(loc)
        plt.savefig(f'static\\{tag}' if operating_system() == 'Windows' else f'static/{tag}')
    except :
        loc = f'img\\{tag}' if operating_system() == 'Windows' else f'img/{tag}'
        plt.savefig(loc)
        plt.savefig(f'src\\static\\{tag}' if operating_system() == 'Windows' else f'src/static/{tag}')

    
    arr = predictor.forecast(365)

    return os.path.abspath(loc), {"1 day": arr[0], "7 days": arr[6], "1 month": arr[29], "6 months": arr[(30 * 6) - 1], "1 year": arr[364]} # TODO add the accuracy

if __name__ == '__main__':
    print(get_predictive_model('VHC', pd.to_datetime('2018-01-01')))

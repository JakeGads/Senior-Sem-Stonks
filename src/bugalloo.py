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

    for i in ['data', 'img']:
        os.system(f'mkdir {i}')
    # forces tag into a list
    tag = [tag]
    # attempts to pull the data
    try:
        # get it from yahoo
        data = pdr.get_data_yahoo(tag, start=start_date, end=end_date)
        
        data.reset_index(inplace=True,drop=True)
        
        data.to_csv(f'data/{tag[0]}.csv')

        with open(f'data/{tag[0]}.csv', 'r') as in_file:
            lines = in_file.readlines()
        with open(f'data/{tag[0]}.csv', 'w+') as out_file:
            del lines[1]
            lines[0] = lines[0].replace('Attributes', 'Date')
            for i in lines:
                out_file.write(i)
        
        data = pd.read_csv(f'data/{tag[0]}.csv')

        # returns the data
        return data

    except :
        # if the data is wrong, return a blank one
        
        return pd.DataFrame()


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler

from platform import system as operating_system

def get_predictive_model(tag:str, start_date = pd.to_datetime('2020-01-01'), end_date = dt.datetime.today()):

    df = get_stock_data(tag, start_date, end_date)

    df.index = df["Date"]
    
    data=df.sort_index(ascending=True,axis=0)
    new_dataset=pd.DataFrame(index=range(0,len(df)),columns=['Date','Close'])
    for i in range(0,len(data)):
        new_dataset["Date"][i]=data['Date'][i]
        new_dataset["Close"][i]=data["Close"][i]

    final_dataset=new_dataset.values
    final_dataset_new = np.empty(np.shape(final_dataset))
    for i in range (0, len(final_dataset)):
        final_dataset_new[i][0] = final_dataset[i][0]
        final_dataset_new[i][1] = final_dataset[i][1]
    
    scaler=MinMaxScaler(feature_range=(0,1))
    final_dataset=new_dataset.values
    train_split = int(len(final_dataset) * .80)
    train_data=final_dataset[0:train_split,:]
    valid_data=final_dataset[train_split:,:]
    new_dataset.index=new_dataset.Date
    new_dataset.drop("Date",axis=1,inplace=True)
    scaler=MinMaxScaler(feature_range=(0,1))
    scaled_data=scaler.fit_transform(final_dataset_new)
    x_train_data,y_train_data=[],[]
    for i in range(60,len(train_data)):
        x_train_data.append(scaled_data[i-60:i,0])
        y_train_data.append(scaled_data[i,0])
        
    x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data)
    x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))

    lstm_model=Sequential()
    lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_data.shape[1],1)))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))
    inputs_data=new_dataset[len(new_dataset)-len(valid_data)-60:].values
    inputs_data=inputs_data.reshape(-1,1)
    try:
        inputs_data=scaler.transform(inputs_data)   
    except :
        None
    lstm_model.compile(loss='mean_squared_error',optimizer='adam')
    lstm_model.fit(x_train_data,y_train_data,epochs=1,batch_size=1,verbose=2)

    x_test=[]
    for i in range(60,inputs_data.shape[0]):
        x_test.append(inputs_data[i-60:i,0])
    x_test=np.float32(np.array(x_test))
    x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    
    predicted_closing_price=lstm_model.predict(x_test)
    try:
        predicted_closing_price=scaler.inverse_transform(predicted_closing_price)
    except:
        None

    train_data=new_dataset[:train_split]
    valid_data=new_dataset[train_split:]
    valid_data['Predictions']=predicted_closing_price
    plt.plot(train_data["Close"])
    plt.plot(valid_data[['Close',"Predictions"]])
    
    loc = f'..\\img\\{tag}' if operating_system() == 'Windows' else f'../img/{tag}'
    try:
        plt.savefig(loc)
        # plt.savefig(f'static\\{tag}' if operating_system() == 'Windows' else f'static/{tag}')
    except :
        loc = f'img\\{tag}' if operating_system() == 'Windows' else f'img/{tag}'
        plt.savefig(loc)
        # plt.savefig(f'src\\static\\{tag}' if operating_system() == 'Windows' else f'src/static/{tag}')

    plt.close()
    return os.path.abspath(loc) 

if __name__ == '__main__':
    print(get_predictive_model('VHC', pd.to_datetime('2018-01-01')))
    print(get_predictive_model('UIS', pd.to_datetime('2018-01-01')))
    print(get_predictive_model('NTDOY', pd.to_datetime('2018-01-01')))
    print(get_predictive_model('GME', pd.to_datetime('2018-01-01')))

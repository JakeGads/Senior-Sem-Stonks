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
        # data = data.drop(["Adj Close", "Volume"], axis = 1)


        # returns the data
        return data
    except :
        # if the data is wrong, return a blank one
        print("errord")
        return pd.DataFrame()


import numpy as np
from tqdm import tqdm as loading_bar
from keras.models import Sequential 
from keras.layers import Dense

def get_predictive_model(tag:str, start_date = pd.to_datetime('2020-01-01'), end_date = dt.datetime.today()):
    df = get_stock_data(tag, start_date, end_date)
    
    y = np.array(df[['Close']])
    X = np.array(df['Date'].apply(lambda x: x.toordinal()))

    # https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
    
    model = Sequential()
    model.add(Dense(12,input_dim=df.shape[1],activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.fit(X, y, epochs=150, batch_size=10)

    model, accuracy = model.evaluate(X, y)
    print('Accuracy: %.2f' % (accuracy*100))

    predictions = model.predict(X)
    # round predictions 
    rounded = [round(x[0]) for x in predictions]

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
    get_predictive_model('gme', pd.to_datetime('2010-01-01'))

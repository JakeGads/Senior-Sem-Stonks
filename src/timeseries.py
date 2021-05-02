import os
import subprocess
from platform import system as operating_system
import threading

from numpy.lib.function_base import average
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

# https://cnvrg.io/pytorch-lstm/

def get_stock_data(tag: str, start_date: dt.datetime, end_date: dt.datetime) -> pd.core.frame.DataFrame:

    for i in ['data', 'img']:
        os.system(f'mkdir {i}')
        if operating_system() == 'Windows':
            os.system('cls')
        else:
            os.system('clear')
    
                
    # forces tag into a list
    tag = [tag]
    # attempts to pull the data
    try:
        # get it from yahoo
        data = pdr.get_data_yahoo(tag, start=start_date, end=end_date)
        # generate a index
        """
        Date,Adj Close,Close,High,Low,Open,Volume
        df = df.reindex(columns=column_names)
        ___
        df = df[['favorite_color','grade','name','age']]
        ___
        df1 = pd.DataFrame(df1,columns=['Name','Gender','Score','Rounded_score'])

        """
        
        # write it out in the og format
        data.to_csv(f'data/{tag[0]}.csv')

        # so that it can be read in 
        with open(f'data/{tag[0]}.csv', 'r') as in_file:
            lines = in_file.readlines()
        
        # and manipulated before being exported
        with open(f'data/{tag[0]}.csv', 'w+') as out_file:
            
            lines[0] = lines[0].replace('Attributes', 'Date')
            del lines[1:3]
            
            for i in lines:
                out_file.write(i)
        
        data = pd.read_csv(f'data/{tag[0]}.csv', index_col=0, parse_dates=True)

        data = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
        # data["str_date"] = data["Date"]
        # data["Date"] = data["Date"].apply(
        #     lambda x: dt.datetime(
        #             *list(
        #                 map(
        #                     int, x.split('-')
        #                 )
        #             )
        #         ).toordinal()
        #     )

        # and loaded in as a dataframe and exported out as the function
        return data

    except Exception as e:
        # if the data is wrong, return a blank one
        print(e)
        return pd.DataFrame()


from flask import url_for

# Mathematical functions 
import math 
# Fundamental package for scientific computing with Python
import numpy as np 
# Additional functions for analysing and manipulating data
import pandas as pd 
# Date Functions
from datetime import date, timedelta, datetime
# This function adds plotting functions for calender dates
from pandas.plotting import register_matplotlib_converters
# Important package for visualization - we use this to plot the market data
import matplotlib.pyplot as plt 
# Formatting dates
import matplotlib.dates as mdates
# Packages for measuring model performance / errors
from sklearn.metrics import mean_absolute_error, mean_squared_error
# Deep learning library, used for neural networks
from keras.models import Sequential 
# Deep learning classes for recurrent and regular densely-connected layers
from keras.layers import LSTM, Dense, Dropout
# EarlyStopping during model training
from keras.callbacks import EarlyStopping
# This Scaler removes the median and scales the data according to the quantile range to normalize the price data 
from sklearn.preprocessing import RobustScaler

def get_predictive_model(tag:str, start_date = pd.to_datetime('2020-01-01'), end_date = dt.datetime.today()):
    # pull in data from 
    
    df = get_stock_data(tag, start_date, end_date)
    train_dfs = df.copy()

    # Indexing Batches
    train_df = train_dfs.sort_values(by=['Date']).copy()

    # We safe a copy of the dates index, before we need to reset it to numbers
    date_index = train_df.index

    # Adding Month and Year in separate columns
    d = pd.to_datetime(train_df.index)
    train_df['Month'] = d.strftime("%m") 
    train_df['Year'] = d.strftime("%Y") 

    # We reset the index, so we can convert the date-index to a number-index
    train_df = train_df.reset_index(drop=True).copy()

    FEATURES = ['High', 'Low', 'Open', 'Close', 'Volume', 'Month']

    data = pd.DataFrame(train_df)
    data_filtered = data[FEATURES]

    # We add a prediction column and set dummy values to prepare the data for scaling
    data_filtered_ext = data_filtered.copy()
    data_filtered_ext['Prediction'] = data_filtered_ext['Close'] 

    # Calculate the number of rows in the data
    nrows = data_filtered.shape[0]
    np_data_unscaled = np.array(data_filtered)
    np_data_unscaled = np.reshape(np_data_unscaled, (nrows, -1))
    

    # Transform the data by scaling each feature to a range between 0 and 1
    scaler = RobustScaler()
    np_data = scaler.fit_transform(np_data_unscaled)

    # Creating a separate scaler that works on a single column for scaling predictions
    scaler_pred = RobustScaler()
    df_Close = pd.DataFrame(data_filtered_ext['Close'])
    np_Close_scaled = scaler_pred.fit_transform(df_Close)

    #Settings
    sequence_length = 100

    # Split the training data into x_train and y_train data sets
    # Get the number of rows to train the model on 80% of the data 
    train_data_len = math.ceil(np_data.shape[0] * 0.8) #2616

    # Create the training data
    train_data = np_data[0:train_data_len, :]
    x_train, y_train = [], []

    # The RNN needs data with the format of [samples, time steps, features].
    # Here, we create N samples, 100 time steps per sample, and 2 features
    for i in range(100, train_data_len):
        x_train.append(train_data[i-sequence_length:i,:]) #contains 100 values 0-100 * columsn
        y_train.append(train_data[i, 0]) #contains the prediction values for validation
        
    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Create the test data
    test_data = np_data[train_data_len - sequence_length:, :]

    # Split the test data into x_test and y_test
    x_test, y_test = [], []
    test_data_len = test_data.shape[0]
    for i in range(sequence_length, test_data_len):
        x_test.append(test_data[i-sequence_length:i,:]) #contains 100 values 0-100 * columns
        y_test.append(test_data[i, 0]) #contains the prediction values for validation
    # Convert the x_train and y_train to numpy arrays
    x_test, y_test = np.array(x_test), np.array(y_test)

    # Convert the x_train and y_train to numpy arrays
    x_test = np.array(x_test); y_test = np.array(y_test)
        

    # Configure the neural network model
    model = Sequential()

    # Model with 100 Neurons 
    # inputshape = 100 Timestamps, each with x_train.shape[2] variables
    n_neurons = x_train.shape[1] * x_train.shape[2]
    
    model.add(LSTM(n_neurons, return_sequences=False, 
                input_shape=(x_train.shape[1], x_train.shape[2]))) 
    model.add(Dense(1, activation='relu'))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    epochs = 5  
    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    history = model.fit(x_train, y_train, batch_size=16, 
                        epochs=epochs, callbacks=[early_stop])

    # Get the predicted values
    predictions = model.predict(x_test)

    # Mean Absolute Percentage Error (MAPE)
    MAPE = np.mean((np.abs(np.subtract(y_test, predictions)/ y_test))) * 100
    

    # Median Absolute Percentage Error (MDAPE)
    MDAPE = np.median((np.abs(np.subtract(y_test, predictions)/ y_test)) ) * 100
    
    # Get the predicted values
    pred_unscaled = scaler_pred.inverse_transform(predictions)

    # The date from which on the date is displayed
    display_start_date = pd.Timestamp('today') - timedelta(days=500)

    # Add the date column
    data_filtered_sub = data_filtered.copy()
    data_filtered_sub['Date'] = date_index

    # Add the difference between the valid and predicted prices
    train = data_filtered_sub[:train_data_len + 1]
    valid = data_filtered_sub[train_data_len:]
    valid.insert(1, "Prediction", pred_unscaled.ravel(), True)
    valid.insert(1, "Difference", valid["Prediction"] - valid["Close"], True)

    # Zoom in to a closer timeframe
    valid = valid[valid['Date'] > display_start_date]
    train = train[train['Date'] > display_start_date]

    # Visualize the data
    plt.subplots(figsize=(10, 8), sharex=True)
    xt = train['Date']; yt = train[["Close"]]
    xv = valid['Date']; yv = valid[["Close", "Prediction"]]
    plt.title("Predictions vs Actual Values", fontsize=20)
    plt.ylabel(tag, fontsize=18)
    plt.plot(xt, yt, color="#039dfc", linewidth=2.0)
    plt.plot(xv, yv["Prediction"], color="#E91D9E", linewidth=2.0)
    plt.plot(xv, yv["Close"], color="black", linewidth=2.0)
    plt.legend(["Train", "Test Predictions", "Actual Values"], loc="upper left")

    # # Create the bar plot with the differences
    x = valid['Date']
    y = valid["Difference"]

    # Create custom color range for positive and negative differences
    valid.loc[y >= 0, 'diff_color'] = "#2BC97A"
    valid.loc[y < 0, 'diff_color'] = "#C92B2B"

    plt.bar(x, y, width=0.8, color=valid['diff_color'])
    plt.grid()           

    save_loc = f'..\\img\\{tag}' if operating_system() == 'Windows' else f'../img/{tag}'

    try:
        plt.savefig(save_loc)
        save_loc = f'static\\{tag}' if operating_system() == 'Windows' else f'static/{tag}'
        plt.savefig(save_loc)
    except :
        save_loc = f'img\\{tag}' if operating_system() == 'Windows' else f'img/{tag}'
        plt.savefig(save_loc)
        save_loc = f'src\\static\\{tag}' if operating_system() == 'Windows' else f'src/static/{tag}'
        plt.savefig(save_loc)
    
    new_df = df

    d = pd.to_datetime(new_df.index)
    new_df['Month'] = d.strftime("%m") 
    new_df['Year'] = d.strftime("%Y") 
    new_df = new_df.filter(FEATURES)

    # Get the last 100 day closing price values and scale the data to be values between 0 and 1
    last_100_days = new_df[-int(len(new_df) * .1):].values
    last_100_days_scaled = scaler.transform(last_100_days)

    # Create an empty list and Append past 100 days
    X_test_new = []
    X_test_new.append(last_100_days_scaled)

    # Convert the X_test data set to a numpy array and reshape the data
    pred_price_scaled = model.predict(np.array(X_test_new))
    pred_price_unscaled = scaler_pred.inverse_transform(pred_price_scaled)
    values = []
    percentages = []
    for i in range(1,7):
        # Print last price and predicted price for the next day
        price_today = round(new_df['Close'][-1], 2)
        predicted_price = round(pred_price_unscaled.ravel()[0], i + 2)
        percent = round(100 - (predicted_price * 100)/price_today, 2)

        a = '+'
        if percent > 0:
            a = '-'

        
        values.append(predicted_price)
        percentages.append(percent)   
    
    
    
    expected = f'The average predicted close price after a week is {round(average(values), 1)}({a}{average(percentages)}%)'
    print(f'finished {tag}')
    try:
        return url_for("static", filename=f'{tag}.png'), expected
    except:
        return save_loc, expected

if __name__ == '__main__':
    for i in ['GME']:
        print(
            get_predictive_model(i, pd.to_datetime('2020-01-01'))
            # dt.datetime.now() - dt.timedelta(days=368))
        )
        
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


from dateutil.parser import parse 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def get_predictive_model(tag:str, start_date = pd.to_datetime('2020-01-01'), end_date = dt.datetime.today()):
    df = get_stock_data(tag, start_date, end_date)
    
    
    
    # plt.figure(figsize = (14,7))
    # plt.plot(range(df.shape[0]),all_mid_data,color='b',label='True')
    # plt.plot(range(0,N),run_avg_predictions,color='orange', label='Prediction')
    # plt.xticks(range(0,df.shape[0],50),df['Date_String'].loc[::50],rotation=45)
    # plt.xlabel('Date')
    # plt.ylabel('Mid Price')
    # plt.legend(fontsize=12)
    # loc = f'..\\img\\{tag}' if operating_system() == 'Windows' else f'../img/{tag}'
    # try:
    #     plt.savefig(loc)
    #     plt.savefig(f'static\\{tag}' if operating_system() == 'Windows' else f'static/{tag}')
    # except :
    #     loc = f'img\\{tag}' if operating_system() == 'Windows' else f'img/{tag}'
    #     plt.savefig(loc)
    #     plt.savefig(f'src\\static\\{tag}' if operating_system() == 'Windows' else f'src/static/{tag}')

    
    # return os.path.abspath(loc)
if __name__ == '__main__':
    print(get_predictive_model('NTDOY', pd.to_datetime('2020-01-01')))

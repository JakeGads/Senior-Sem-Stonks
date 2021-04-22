from pandas_datareader import data as pdr
import pandas as pd

import numpy as np 
import torch
import matplotlib.pyplot as plt

import os
from platform import system as operating_system
from math import ceil
import datetime as dt
from tqdm import tqdm

from torch.utils.data import Dataset
from torch.utils.data import DataLoader 

from torch import nn

class timeseries(Dataset):
    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32)
        self.len = x.shape[0]

    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
  
    def __len__(self):
        return self.len

class neural_network(nn.Module):
    def __init__(self):
        super(neural_network,self).__init__()
        self.lstm = nn.LSTM(input_size=1,hidden_size=5,num_layers=1,batch_first=True)
        self.fc1 = nn.Linear(in_features=5,out_features=1)

    def forward(self,x):
        output,_status = self.lstm(x)
        output = output[:,-1,:]
        output = self.fc1(torch.relu(output))
        return output

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
        # data = data.drop(["Adj Close", "Volume"], axis = 1)


        # returns the data
        return data
    except :
        # if the data is wrong, return a blank one
        print("errord")
        return pd.DataFrame()

def get_predictive_model(tag:str, start_date = pd.to_datetime('2020-01-01'), end_date = dt.datetime.today()):
    df = get_stock_data(tag, start_date, end_date)

    y = np.array(df[['Close']])
    x = np.array(df['Date'].apply(lambda x: x.toordinal()))

    x_train = np.array(x[int(len(x) * .75):])
    x_test = np.array(x[:int(len(y) * .75)])

    y_train = np.array(y[int(len(y) * .75):])
    y_test =  np.array(y[:int(len(y) * .75)])
    
    dataset = timeseries(x,y)
    train_loader = DataLoader(dataset,shuffle=True,batch_size=100)

    model = neural_network()

    # optimizer , loss
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
    epochs = 1500

    # finding a good shape size

    size = 1,1

    def find_size(x:int):
        for i in range(1, ceil(x/2)):
            for j in range(1, ceil(x/2)):
                if i * j == x:
                    return x
                    # return max([i,j])
        
        return x

    for i in tqdm(range(epochs)):
        for j,data in enumerate(train_loader):
            size = find_size(len(data[:][0]))
            try:
                y_pred = model(data[:][0].view(1,size,1)).reshape(-1)
            except:
                print('size:', size)
                continue

            loss = criterion(y_pred,data[:][1])
            loss.backward()
            optimizer.step()
        

    test_set = timeseries(x_test,y_test)
    test_pred = model(test_set[:][0].view(-1,1,1)).view(-1)
    plt.plot(test_pred.detach().numpy(),label='predicted')
    plt.plot(test_set[:][1].view(-1),label='original')
    
    plt.legend()
    plt.show()
    
    
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
    get_predictive_model('gme', pd.to_datetime('2015-01-01'))

import pandas as pd
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf 
import datetime

def generate_stock_data(tag, start_date = pd.to_datetime('2017-01-01'), end_date = datetime.datetime.today()):
    tag = [tag]
    try:
        data = pdr.get_data_yahoo(tag, start=start_date, end=end_date)
    except :
        return 0


if __name__ == '__main__':
    generate_stock_data('gamer')
from flask import Flask, render_template, url_for, redirect
from timeseries import get_predictive_model
from datetime import datetime as dt
import concurrent.futures
from pandas import to_datetime

app = Flask(__name__)

class tag_data:
    def __init__(self, tag, start, end, data):
        self.tag = tag
        self.start = start
        self.end = end
        self.img = url_for('static', filename=self.tag + '.png')
        self.data = data

    def __repr__(self):
        return f'''{self.tag}
            {self.img}
            {self.data}
            '''
        
    
    def __str__(self):
        return self.__repr__()

@app.route('/')
@app.route('/home/')
@app.route('/home/<tags>')
@app.route('/home/<tags>/<start_date>')
def home(tags:str = "GME", start_date: str = to_datetime("2018-01-01"), end_date: str = dt.now()):
    print(f'/home/{tags}/{start_date}')
    if type(start_date) == type('string'):
        start_date = to_datetime(start_date)
    if type(end_date) == type('string'):
        end_date = to_datetime(end_date)
    data = []
    tags = tags.split('|')
    for i in tags:
        try:
            # thread = ''
            # with concurrent.futures.ThreadPoolExecutor() as executor:
            #     thread = executor.submit(get_predictive_model, [i, start_date, end_date])
            data.append(
                tag_data(i, 
                    start_date, 
                    end_date, 
                    get_predictive_model(i, start_date, end_date)
                )
            )
        except:
            data.append(tag_data(i, start_date, end_date, ['error.png', 'error  ']))
    for i in data:
        print(i)

    

    return render_template('index.html', length=len(data), data=data)

if __name__ == '__main__':
    app.run(host ='0.0.0.0', port = 5001, debug = True)

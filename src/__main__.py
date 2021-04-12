from flask import Flask, render_template
from predective_model import clear_static, predective_model
from datetime import datetime as dt
app = Flask(__name__)

class tag_data:
    def __init__(self, tag, start, end, data):
        self.tag = tag
        self.start = start
        self.end = end
        self.data = data


@app.route('/')
def index():
    home()

@app.route('/index/<tags>/<start_date>/<end_date>')
def home(tags:list = [], start_date: dt = dt.now(), end_date: dt = dt.now()):
    clear_static()
    
    data = []

    for i in tags.split(','):
        data.append(tag_data(i, start_date, end_date, predective_model(i, start_date. end_date)))
        
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
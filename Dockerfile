FROM python:3.7

ADD main.py .

RUN python -m pip install --upgrade pip

RUN pip install coin

CMD [ "python", "./main.py"]

#To build
#docker build -t python-stonks

#To run without user input
#docker run python-stonks

#To run with user input
#docker run -t -i python-stonks
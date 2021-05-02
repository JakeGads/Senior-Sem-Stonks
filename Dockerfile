FROM python:3.7

ADD src/__main__.py .
ADD src/model.py .
ADD src/timeseries.py .
ADD req.pip .

EXPOSE 5000/udp
EXPOSE 5000/tcp

RUN python -m pip install --upgrade pip

RUN pip install -r req.pip

CMD [ "python", "./__main__.py"]

#To build
#docker build -t python-stonks .

#To run without user input
#docker run python-stonks

#To run with user input
#docker run -t -i python-stonks
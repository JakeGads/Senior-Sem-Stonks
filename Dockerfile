# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster
WORKDIR /app
COPY /src /app
COPY requirements.txt /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 5001
ENTRYPOINT [ "python" ]
CMD [ "__main__.py" ]


#To build
#docker build -t python-stonks .

#To run without user input
#docker run python-stonks

#To run with user input
#docker run -t -i python-stonks

#To run with ports turned on
# docker run -d -p 5000:5000 python-stonks
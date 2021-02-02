FROM python:3.7.5-slim

ENV VIRTUAL_ENV "/venv"
RUN python -m venv $VIRTUAL_ENV
ENV PATH "$VIRTUAL_ENV/bin:$PATH"
RUN python -m pip install --upgrade pip
RUN python -m pip install \
        certifi \
        chardet \
        fix \
        idna    \
        lxml    \
        multitasking    \
        numpy   \
        pandas  \
        pandas-datareader \
        python-dateutil \
        pytz    \
        requests    \
        six \
        urllib3 \
        tensorflow

COPY src/* src/

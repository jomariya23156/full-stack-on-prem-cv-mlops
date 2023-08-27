FROM python:3.9.17-slim

ARG WEB_UI_PORT=$WEB_UI_PORT

WORKDIR /service

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY app/ app/

EXPOSE $WEB_UI_PORT

WORKDIR /service/app

CMD uvicorn main:app --host 0.0.0.0 --port ${WEB_UI_PORT}
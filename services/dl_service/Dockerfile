FROM python:3.9.17-slim

ARG DL_SERVICE_PORT=$DL_SERVICE_PORT

WORKDIR /service

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY app/ app/

EXPOSE $DL_SERVICE_PORT

WORKDIR /service/app

CMD gunicorn main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:${DL_SERVICE_PORT}
FROM python:3.9.17-slim

ARG PREFECT_PORT=$PREFECT_PORT

COPY requirements.txt .

RUN pip install -r requirements.txt

EXPOSE $PREFECT_PORT

CMD prefect server start --host 0.0.0.0 --port ${PREFECT_PORT}
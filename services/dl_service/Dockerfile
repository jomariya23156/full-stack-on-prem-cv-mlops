FROM python:3.9.17-slim

ARG DL_SERVICE_PORT=$DL_SERVICE_PORT

# for using curl in health check
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl \
        tzdata \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /service

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY app/ app/

EXPOSE $DL_SERVICE_PORT

WORKDIR /service/app

CMD gunicorn main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:${DL_SERVICE_PORT}
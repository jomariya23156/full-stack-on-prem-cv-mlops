FROM python:3.9.17-slim

ARG PREFECT_PORT=$PREFECT_PORT

# for using curl in health check
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl \
        tzdata \
        build-essential \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install -r requirements.txt

EXPOSE $PREFECT_PORT

CMD prefect server start --host 0.0.0.0 --port ${PREFECT_PORT}
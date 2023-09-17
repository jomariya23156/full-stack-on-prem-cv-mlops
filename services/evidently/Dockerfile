FROM python:3.9.17-slim

ARG EVIDENTLY_PORT=$EVIDENTLY_PORT

# for time zone data
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        tzdata \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /service

COPY requirements.txt .

RUN pip install -r requirements.txt

EXPOSE $EVIDENTLY_PORT

CMD evidently ui --workspace ${EVIDENTLY_WORKSPACE_NAME}/ --port ${EVIDENTLY_PORT}
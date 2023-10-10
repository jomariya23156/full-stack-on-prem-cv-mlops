FROM python:3.9.17-slim

WORKDIR /service

# for using git-based deployment
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git \
        tzdata \
        build-essential \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install -r requirements.txt

CMD prefect worker start --pool ${PREFECT_MONITOR_WORK_POOL} --type process
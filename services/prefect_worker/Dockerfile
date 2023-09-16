FROM python:3.9.17-slim

WORKDIR /workspace

# for using curl in health check
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install -r requirements.txt

CMD prefect worker start --pool production-model-pool --type process
#! /bin/bash

# uvicorn main:app --host 0.0.0.0 --port 4242 --reload

gunicorn main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:4242
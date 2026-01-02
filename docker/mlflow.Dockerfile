FROM python:3.10-slim

RUN pip install --no-cache-dir mlflow

CMD mlflow server \
    --backend-store-uri file:///mlruns \
    --default-artifact-root /mlruns \
    --host 0.0.0.0 \
    --port 5000

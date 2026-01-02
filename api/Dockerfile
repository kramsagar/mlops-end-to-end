FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi uvicorn mlflow numpy

COPY api/ api/
COPY src/logger.py src/logger.py

ENV MLFLOW_TRACKING_URI=file:///mlruns

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

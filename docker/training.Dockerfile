FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    scikit-learn pandas mlflow dvc

COPY src/ src/
COPY data/ /data/

ENV MLFLOW_TRACKING_URI=file:///mlruns

CMD ["python", "src/train.py"]

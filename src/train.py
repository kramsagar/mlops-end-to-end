import os
import hashlib
import subprocess
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
from logger import get_logger

logger = get_logger("training")

DATA_PATH = "/data/data.csv"
MLRUNS = "/mlruns"

def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except:
        return "no-git"

mlflow.set_tracking_uri("file://" + MLRUNS)
mlflow.set_experiment("linear-regression")

df = pd.read_csv(DATA_PATH)
X = df[["x"]]
y = df["y"]

model = LinearRegression()
model.fit(X, y)

preds = model.predict(X)
mse = mean_squared_error(y, preds)

dataset_hash = file_hash(DATA_PATH)

with mlflow.start_run() as run:
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_metric("mse", mse)
    mlflow.log_param("dataset_hash", dataset_hash)
    mlflow.log_param("git_commit", git_commit())

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="LinearRegressionModel"
    )

    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions("LinearRegressionModel")
    version = versions[-1].version

    client.transition_model_version_stage(
        name="LinearRegressionModel",
        version=version,
        stage="Production",
        archive_existing_versions=True
    )

logger.info("training_complete")

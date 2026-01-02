import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from utils import file_hash, git_commit
import os

DATA_PATH = "data/raw/data.csv"
EXPERIMENT = "linear-regression-local"

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment(EXPERIMENT)

df = pd.read_csv(DATA_PATH)
X = df[["x"]]
y = df["y"]

model = LinearRegression()

with mlflow.start_run() as run:
    model.fit(X, y)
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)

    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mse", mse)
    mlflow.log_param("dataset_hash", file_hash(DATA_PATH))
    mlflow.log_param("git_commit", git_commit())

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="LinearRegressionModel"
    )

    print("RUN_ID:", run.info.run_id)

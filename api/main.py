import mlflow
import time
import logging
from fastapi import FastAPI
from pydantic import BaseModel

mlflow.set_tracking_uri("file:/mlruns")

model = mlflow.pyfunc.load_model(
    model_uri="models:/LinearRegressionModel/Production"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

app = FastAPI()

class Input(BaseModel):
    x: float

@app.post("/predict")
def predict(data: Input):
    start = time.time()
    y_pred = model.predict([[data.x]])[0]
    latency = time.time() - start

    logging.info({
        "x": data.x,
        "prediction": y_pred,
        "latency_ms": latency * 1000,
        "model_version": "Production"
    })

    return {
        "prediction": y_pred,
        "latency_ms": latency * 1000
    }

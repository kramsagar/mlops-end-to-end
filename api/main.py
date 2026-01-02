from fastapi import FastAPI
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import mlflow
import time
import logging
from pydantic import BaseModel

mlflow.set_tracking_uri("file:/mlruns")

model = mlflow.pyfunc.load_model(
    "models:/LinearRegressionModel/Production"
)

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

class Input(BaseModel):
    x: float

@app.post("/predict")
def predict(i: Input):
    start = time.time()
    y = model.predict([[i.x]])[0]
    latency = (time.time() - start) * 1000

    logging.info({
        "x": i.x,
        "prediction": y,
        "latency_ms": latency,
        "model": "LinearRegressionModel:Production"
    })

    return {
        "prediction": y,
        "latency_ms": latency
    }

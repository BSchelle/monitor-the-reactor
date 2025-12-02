from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from preprocess_predict import PredictionPipeline


# Initialization of the prediction pipeline
pipeline = PredictionPipeline(
    "models/best_model.pkl",
    "models/preprocessor.pkl"
)

app = FastAPI(
    title="Reactor Guardian API",
    description="API de maintenance prédictive pour le réacteur industriel",
    version="1.0.0",
)


# prediction input data model
class InputData(BaseModel):
    temperature: float
    pressure: float
    flow_rate: float
    vibration: float
    threshold: Optional[float] = 0.8  # configurable alert threshold


@app.get("/")
def root():
    """
    Endpoint racine, utilisé pour vérifier que l'API tourne.
    """
    return {"greeting": "Reactor Guardian API is running"}


@app.get("/health")
def health():
    """
    Endpoint de santé de l'API.
    """
    return {
        "status": "ok",
        "model_loaded": True,
        "details": "API running and model loaded",
    }


@app.post("/predict")
def predict(data: InputData):
    """
    Endpoint de prédiction.
    Reçoit les données capteurs et renvoie la probabilité de panne + alerte.
    """
    payload = data.dict()
    threshold = payload.pop("threshold", 0.8)

    result = pipeline.predict_with_alert(payload, threshold=threshold)

    return {
        "probability": result["probability"],
        "alert": result["alert"],
        "threshold": result["threshold"],
    }

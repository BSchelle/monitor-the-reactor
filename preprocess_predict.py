import joblib
import numpy as np
from pathlib import Path


class PredictionPipeline:
    def __init__(self, model_path: str, scaler_path: str):
        model_path = Path(model_path)
        scaler_path = Path(scaler_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Modèle introuvable : {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler introuvable : {scaler_path}")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        # dataset features
        self.features_order = [
            "temperature",
            "pressure",
            "flow_rate",
            "vibration",
            # features
        ]

    def preprocess(self, data_dict: dict) -> np.ndarray:
        missing = [f for f in self.features_order if f not in data_dict]
        if missing:
            raise ValueError(f"Features manquantes : {missing}")

        values = [data_dict[f] for f in self.features_order]
        X = np.array(values).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        return X_scaled

    def predict_proba(self, data_dict: dict) -> float:
        X_scaled = self.preprocess(data_dict)
        proba = self.model.predict_proba(X_scaled)[0, 1]
        return float(proba)

    def predict_with_alert(self, data_dict: dict, threshold: float = 0.8) -> dict:
        proba = self.predict_proba(data_dict)
        alert = proba >= threshold
        return {
            "probability": proba,
            "alert": alert,
            "threshold": threshold,
        }

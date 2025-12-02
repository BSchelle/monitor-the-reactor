from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_root_is_up():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "greeting" in data


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_predict_endpoint_returns_probability_and_alert():
    payload = {
        "temperature": 120.0,
        "pressure": 2.5,
        "flow_rate": 150.0,
        "vibration": 0.8,
        "threshold": 0.8
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "probability" in data
    assert "alert" in data
    assert 0.0 <= data["probability"] <= 1.0

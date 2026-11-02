from fastapi.testclient import TestClient

from fraudspotter.api.app import create_app
from fraudspotter.api.routes import set_active_model


def test_api_health_endpoint():
    app = create_app()
    client = TestClient(app)
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_api_predict_and_explain(trained_dummy_classical_pipeline):
    app = create_app()
    client = TestClient(app)

    # Set mock pipeline
    set_active_model(trained_dummy_classical_pipeline, "classical")

    payload = {
        "title": "Unbelievable Breakthrough In Science",
        "text": "Secret formula discovered to cure everything without medicine.",
    }

    # Predict test
    pred_res = client.post("/api/v1/predict", json=payload)
    assert pred_res.status_code == 200
    pdata = pred_res.json()
    assert pdata["label"] in ["FAKE", "REAL"]
    assert 0.0 <= pdata["probability_fake"] <= 1.0
    assert 0.0 <= pdata["probability_real"] <= 1.0

    # Explain test
    exp_res = client.post("/api/v1/explain", json=payload)
    assert exp_res.status_code == 200
    edata = exp_res.json()
    assert "prediction" in edata
    assert "attributions" in edata

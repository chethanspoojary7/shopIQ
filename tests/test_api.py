import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app import app

client = app.test_client()

def test_health_shape():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert "model_loaded" in r.get_json()

def test_predict_rejects_bad_input():
    r = client.post("/api/predict", json={"session": {"Month": "January"}})
    assert r.status_code == 400

def test_predict_rejects_missing_session():
    r = client.post("/api/predict", json={})
    assert r.status_code == 400

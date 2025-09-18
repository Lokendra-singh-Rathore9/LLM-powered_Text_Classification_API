import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_classify_endpoint():
    response = client.post("/classify", json={"text": "This is a test message"})
    assert response.status_code == 200
    data = response.json()
    assert "class" in data
    assert "confidence" in data
    assert "prompt_used" in data
    assert "latency_ms" in data

def test_feedback_endpoint():
    response = client.post("/feedback", json={
        "text": "test", 
        "predicted": "safe", 
        "correct": "safe"
    })
    assert response.status_code == 200

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200

def test_health_endpoint():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
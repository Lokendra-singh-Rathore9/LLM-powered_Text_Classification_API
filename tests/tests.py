import sys
import os
import pytest
from fastapi.testclient import TestClient

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from app.main import app

client = TestClient(app)

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_classify_safe():
    """Test classification of safe content"""
    response = client.post("/classify", json={
        "text": "This is a great product! I highly recommend it."
    })
    assert response.status_code == 200
    data = response.json()
    assert "class" in data
    assert data["class"] in ["toxic", "spam", "safe"]
    assert "prompt_used" in data
    assert "latency_ms" in data
    assert isinstance(data["latency_ms"], int)

def test_classify_toxic():
    """Test classification of toxic content"""
    response = client.post("/classify", json={
        "text": "You are stupid and should go die"
    })
    assert response.status_code == 200
    data = response.json()
    assert "class" in data

def test_classify_spam():
    """Test classification of spam content"""
    response = client.post("/classify", json={
        "text": "Buy now! Limited time offer! Click here!"
    })
    assert response.status_code == 200
    data = response.json()
    assert "class" in data

def test_feedback_endpoint():
    """Test feedback submission"""
    response = client.post("/feedback", json={
        "text": "test message", 
        "predicted": "safe", 
        "correct": "spam"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "feedback recorded"

def test_metrics_endpoint():
    """Test metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "class_distribution" in data
    assert "feedback_counts" in data
    assert "latency" in data

if __name__ == "__main__":

    print("Running tests...")
    
    # Run tests manually
    test_health_check()
    print("✅ Health check passed")
    
    test_classify_safe()
    print("✅ Safe classification passed")
    
    test_feedback_endpoint()
    print("✅ Feedback endpoint passed")
    
    test_metrics_endpoint()
    print("✅ Metrics endpoint passed")
    
    print("🎉 All tests passed!")
"""
Unit tests for FastAPI endpoints in api.py.
"""

from fastapi.testclient import TestClient
import io
from PIL import Image
from api import app

client = TestClient(app)


def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "Krishi Rakshak AI"
    assert data["team"] == "Team Codex Verse"
    assert data["status"] == "online"


def test_api_diseases():
    response = client.get("/diseases?lang=en")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 10
    assert "Potato___Early_blight" in data["diseases"]


def test_api_weather():
    response = client.get("/weather?latitude=28.6139&longitude=77.2090")
    assert response.status_code == 200
    data = response.json()
    assert "weather" in data
    assert "disease_susceptibility_alerts" in data


def test_api_predict():
    # Create synthetic leaf test image
    img = Image.new("RGB", (224, 224), color=(50, 150, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("test_leaf.jpg", buf, "image/jpeg")}
    response = client.post("/predict?lang=en", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "diagnosis" in data
    assert "confidence_level" in data["prediction"]
    assert "scan_id" in data


def test_api_history():
    response = client.get("/history")
    assert response.status_code == 200
    data = response.json()
    assert "analytics" in data
    assert "recent_scans" in data

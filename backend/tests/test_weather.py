"""
Unit tests for WeatherAdvisor and disease risk assessment.
"""

from src.weather import WeatherAdvisor
from src.recommendations import RemedyEngine


def test_weather_disease_risk_rules():
    advisor = WeatherAdvisor()
    engine = RemedyEngine()
    disease_info = engine.get_all_supported_diseases()

    # Simulate cool, high-humidity weather conducive to Potato Late Blight (12-20C, >85% humidity)
    mock_weather = {
        "current": {
            "temperature": 16.0,
            "humidity": 92.0,
            "precipitation": 1.2,
            "wind_speed": 4.0
        }
    }

    alerts = advisor.evaluate_disease_risks(mock_weather, disease_info)
    assert len(alerts) > 0

    # Ensure Potato Late Blight or Tomato Late Blight is detected in alerts
    alert_classes = [a["class_name"] for a in alerts]
    assert any("Late_blight" in c for c in alert_classes)
    assert alerts[0]["risk_level"] == "High"


def test_weather_offline_graceful_handling():
    # Use invalid host to test network failure fallback
    advisor = WeatherAdvisor(timeout=0.01)
    advisor.base_url = "https://invalid-non-existent-domain-12345.xyz"

    result = advisor.get_weather(latitude=28.6139, longitude=77.2090)
    assert result["success"] is False
    assert result["offline"] is True
    assert "current" in result
    assert result["current"]["temperature"] == 25.0

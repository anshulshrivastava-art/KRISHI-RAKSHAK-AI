"""
Unit tests for ScanHistoryDB SQLite storage and analytics.
"""

from pathlib import Path
import pytest
from src.database import ScanHistoryDB


@pytest.fixture
def temp_db(tmp_path: Path):
    db_file = tmp_path / "test_krishi.db"
    return ScanHistoryDB(db_path=db_file)


def test_record_and_retrieve_scan(temp_db):
    scan_id = temp_db.record_scan(
        crop_name="Tomato",
        disease_name="Early Blight",
        is_healthy=False,
        confidence=0.88,
        confidence_level="High",
        weather_temp=26.5,
        weather_humidity=82.0,
        notes="Test scan"
    )
    assert scan_id == 1

    records = temp_db.get_recent_scans(limit=10)
    assert len(records) == 1
    assert records[0]["crop_name"] == "Tomato"
    assert records[0]["disease_name"] == "Early Blight"
    assert records[0]["is_healthy"] == 0
    assert records[0]["confidence"] == 0.88


def test_analytics_computation(temp_db):
    temp_db.record_scan("Potato", "Healthy", True, 0.95, "High")
    temp_db.record_scan("Potato", "Late Blight", False, 0.89, "High")
    temp_db.record_scan("Tomato", "Bacterial Spot", False, 0.65, "Medium")

    analytics = temp_db.get_analytics()
    assert analytics["total_scans"] == 3
    assert analytics["healthy_count"] == 1
    assert analytics["diseased_count"] == 2
    assert analytics["confidence_breakdown"]["High"] == 2
    assert analytics["confidence_breakdown"]["Medium"] == 1
    assert len(analytics["top_diseases"]) >= 1


def test_clear_history(temp_db):
    temp_db.record_scan("Corn", "Common Rust", False, 0.90, "High")
    assert temp_db.get_analytics()["total_scans"] == 1
    
    temp_db.clear_history()
    assert temp_db.get_analytics()["total_scans"] == 0
    assert len(temp_db.get_recent_scans()) == 0

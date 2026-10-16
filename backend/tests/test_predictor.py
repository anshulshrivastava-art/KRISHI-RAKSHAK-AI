"""
Unit tests for KrishiPredictor inference and confidence thresholding.
"""

from PIL import Image
from src.predictor import KrishiPredictor, PredictionResult
import config


def test_predictor_returns_structured_result():
    predictor = KrishiPredictor()
    test_img = Image.new("RGB", (256, 256), color=(40, 120, 40))
    
    result = predictor.predict(test_img)
    
    assert isinstance(result, PredictionResult)
    assert result.class_name in config.SUPPORTED_CLASSES or "Class" in result.class_name
    assert 0.0 <= result.confidence <= 1.0
    assert result.confidence_level in {"High", "Medium", "Low"}
    assert len(result.top_predictions) == 3
    assert "%" in result.confidence_percentage


def test_confidence_level_categorization():
    high_t = config.CONFIDENCE_THRESHOLDS["high"]
    med_t = config.CONFIDENCE_THRESHOLDS["medium"]
    
    # Assert config values are sensible
    assert high_t > med_t
    assert high_t == 0.75
    assert med_t == 0.45


def test_prediction_warnings_on_uncertainty():
    predictor = KrishiPredictor()
    test_img = Image.new("RGB", (224, 224), color=(80, 80, 80))
    result = predictor.predict(test_img)
    
    if result.confidence_level in ["Medium", "Low"]:
        assert result.warning_en is not None
        assert result.warning_hi is not None

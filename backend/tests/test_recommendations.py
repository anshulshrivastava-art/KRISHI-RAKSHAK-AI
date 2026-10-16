"""
Unit tests for RemedyEngine knowledge base and multilingual recommendation retrieval.
"""

from src.recommendations import RemedyEngine
import config


def test_remedy_engine_loads_all_supported_classes():
    engine = RemedyEngine()
    all_diseases = engine.get_all_supported_diseases()
    
    for cls_name in config.SUPPORTED_CLASSES:
        assert cls_name in all_diseases, f"Missing class in knowledge base: {cls_name}"


def test_diagnosis_multilingual_content():
    engine = RemedyEngine()
    
    # Test English retrieval
    en_diag = engine.get_diagnosis("Potato___Late_blight", lang="en")
    assert en_diag["crop_name"] == "Potato"
    assert en_diag["disease_name"] == "Late Blight"
    assert en_diag["is_healthy"] is False
    assert len(en_diag["symptoms"]) > 0
    assert len(en_diag["treatment"]["chemical_guidance"]) > 0
    
    # Test Hindi retrieval
    hi_diag = engine.get_diagnosis("Potato___Late_blight", lang="hi")
    assert hi_diag["crop_name"] == "आलू"
    assert hi_diag["disease_name"] == "पछेती झुलसा"
    assert hi_diag["is_healthy"] is False
    assert len(hi_diag["symptoms"]) > 0
    assert "पत्तियों" in hi_diag["symptoms"][0]


def test_healthy_class_handling():
    engine = RemedyEngine()
    diag = engine.get_diagnosis("Tomato___healthy", lang="en")
    
    assert diag["is_healthy"] is True
    assert diag["severity"] == "None"
    assert "No chemical spray required" in diag["treatment"]["chemical_guidance"][0]


def test_unrecognized_class_fallback():
    engine = RemedyEngine()
    diag = engine.get_diagnosis("AlienCrop___Unknown_Rot", lang="en")
    
    assert diag["status"] == "uncertain"
    assert diag["crop_name"] == "Unknown Crop"

"""
Krishi Rakshak AI - Configuration Module
Team: Codex Verse
Description: Central configuration for dataset paths, model parameters,
             confidence thresholds, supported crops/diseases, and weather locations.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
SAMPLES_DIR = DATA_DIR / "samples"
MODELS_DIR = BASE_DIR / "models"

# Model & Artifact Files
MODEL_PATH = MODELS_DIR / "krishi_rakshak_mobilenetv2.keras"
MODEL_H5_PATH = MODELS_DIR / "krishi_rakshak_mobilenetv2.h5"
CLASS_INDICES_PATH = MODELS_DIR / "class_indices.json"
EVALUATION_REPORT_PATH = MODELS_DIR / "evaluation_report.json"
CONFUSION_MATRIX_PATH = MODELS_DIR / "confusion_matrix.png"
METRICS_SUMMARY_PATH = MODELS_DIR / "metrics_summary.json"

# Knowledge Base & Localization Files
DISEASE_INFO_PATH = BASE_DIR / "disease_info.json"
TRANSLATIONS_PATH = BASE_DIR / "translations.json"
if not TRANSLATIONS_PATH.exists() and (BASE_DIR.parent / "frontend" / "translations.json").exists():
    TRANSLATIONS_PATH = BASE_DIR.parent / "frontend" / "translations.json"
DB_PATH = BASE_DIR / "krishi_rakshak.db"

# ML & Training Hyperparameters
IMG_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 16  # Optimized for 8GB RAM laptops with CPU execution
EPOCHS = 5
LEARNING_RATE = 1e-3
FINE_TUNE_EPOCHS = 3
FINE_TUNE_LEARNING_RATE = 1e-4

# Confidence Classification Thresholds
# High: >= 75% -> Display prediction with high confidence
# Medium: 45% - 74% -> Display result as tentative, suggest rescanning
# Low: < 45% -> Mark as uncertain, advise consulting an agricultural officer (KVK)
CONFIDENCE_THRESHOLDS = {
    "high": 0.75,
    "medium": 0.45
}

# Supported Classes across Indian crops (PlantVillage standard subset)
# 10 core classes covering 3 major crops: Potato, Tomato, Corn (Maize)
SUPPORTED_CLASSES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy"
]

# PlantVillage folder aliases in spMohanty/PlantVillage-Dataset
PLANTVILLAGE_FOLDER_MAP = {
    "Potato___Early_blight": "Potato___Early_blight",
    "Potato___Late_blight": "Potato___Late_blight",
    "Potato___healthy": "Potato___healthy",
    "Tomato___Bacterial_spot": "Tomato___Bacterial_spot",
    "Tomato___Early_blight": "Tomato___Early_blight",
    "Tomato___Late_blight": "Tomato___Late_blight",
    "Tomato___healthy": "Tomato___healthy",
    "Corn___Common_rust": "Corn_(maize)___Common_rust_",
    "Corn___Northern_Leaf_Blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn___healthy": "Corn_(maize)___healthy"
}

# Predefined Indian Agricultural Hubs for Weather Advisory
AGRICULTURAL_HUBS = {
    "New Delhi (NCR)": {"lat": 28.6139, "lon": 77.2090, "state": "Delhi"},
    "Bhopal (Malwa Plateau)": {"lat": 23.2599, "lon": 77.4126, "state": "Madhya Pradesh"},
    "Lucknow (Gangetic Plains)": {"lat": 26.8467, "lon": 80.9462, "state": "Uttar Pradesh"},
    "Chandigarh (Punjab-Haryana Hub)": {"lat": 30.7333, "lon": 76.7794, "state": "Punjab"},
    "Patna (Bihar Alluvial Region)": {"lat": 25.5941, "lon": 85.1376, "state": "Bihar"},
    "Bengaluru (Deccan Plateau)": {"lat": 12.9716, "lon": 77.5946, "state": "Karnataka"},
    "Hyderabad (Telangana Semi-Arid)": {"lat": 17.3850, "lon": 78.4867, "state": "Telangana"},
    "Jaipur (Eastern Rajasthan)": {"lat": 26.9124, "lon": 75.7873, "state": "Rajasthan"},
    "Pune (Western Maharashtra)": {"lat": 18.5204, "lon": 73.8567, "state": "Maharashtra"},
    "Kolkata (Lower Gangetic Plains)": {"lat": 22.5726, "lon": 88.3639, "state": "West Bengal"}
}

# Weather API Endpoint
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_TIMEOUT = 8.0  # seconds

# Helpline Information
KISAN_CALL_CENTRE = "1800-180-1551"  # Toll Free Government of India
PROJECT_NAME = "Krishi Rakshak AI"
TEAM_NAME = "Team Codex Verse"
VERSION = "1.0.0 (MVP)"

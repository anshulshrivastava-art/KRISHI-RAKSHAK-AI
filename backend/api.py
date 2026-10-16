"""
Krishi Rakshak AI - FastAPI Inference & Decision Support Server
Team: Codex Verse
Description: REST API for crop leaf disease detection, weather risk advisory, and farmer history.
"""

from typing import Optional
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from src.preprocessing import ImageValidationError
from src.predictor import KrishiPredictor
from src.recommendations import RemedyEngine
from src.weather import WeatherAdvisor
from src.database import ScanHistoryDB

app = FastAPI(
    title="Krishi Rakshak AI API",
    description="AI-Powered Crop Disease & Pest Assistance API for Indian Farmers by Team Codex Verse",
    version=config.VERSION
)

# Enable CORS for web and mobile frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize singletons
predictor = KrishiPredictor()
remedy_engine = RemedyEngine()
weather_advisor = WeatherAdvisor()
history_db = ScanHistoryDB()


@app.get("/")
def root():
    return {
        "project": config.PROJECT_NAME,
        "team": config.TEAM_NAME,
        "version": config.VERSION,
        "status": "online",
        "demo_mode": predictor.is_demo_mode(),
        "supported_classes_count": len(config.SUPPORTED_CLASSES),
        "docs_url": "/docs"
    }


@app.post("/predict")
async def predict_crop_leaf(
    file: UploadFile = File(...),
    lang: str = Query("en", description="Preferred language: 'en' or 'hi'"),
    latitude: Optional[float] = Query(None, description="Optional field latitude for weather correlation"),
    longitude: Optional[float] = Query(None, description="Optional field longitude for weather correlation")
):
    """
    Upload a leaf image (JPG/PNG), run MobileNetV2 inference,
    retrieve actionable remedies, and record the scan in local history.
    """
    try:
        image_bytes = await file.read()
        prediction = predictor.predict(image_bytes)
    except ImageValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    # Retrieve remedies & symptoms
    diagnosis = remedy_engine.get_diagnosis(prediction.class_name, lang=lang)

    # Optional Weather Correlation
    weather_data = None
    weather_alerts = []
    temp_val = None
    humidity_val = None
    
    if latitude is not None and longitude is not None:
        weather_data = weather_advisor.get_weather(latitude, longitude)
        if weather_data.get("current"):
            temp_val = weather_data["current"]["temperature"]
            humidity_val = weather_data["current"]["humidity"]
            weather_alerts = weather_advisor.evaluate_disease_risks(
                weather_data,
                remedy_engine.get_all_supported_diseases()
            )

    # Save to local SQLite database
    scan_id = history_db.record_scan(
        crop_name=diagnosis["crop_name"],
        disease_name=diagnosis["disease_name"],
        is_healthy=diagnosis["is_healthy"],
        confidence=prediction.confidence,
        confidence_level=prediction.confidence_level,
        weather_temp=temp_val,
        weather_humidity=humidity_val,
        notes=f"Scanned via API - Demo={prediction.is_demo_mode}"
    )

    return {
        "scan_id": scan_id,
        "prediction": {
            "class_name": prediction.class_name,
            "confidence": prediction.confidence,
            "confidence_percentage": prediction.confidence_percentage,
            "confidence_level": prediction.confidence_level,
            "top_predictions": prediction.top_predictions,
            "is_demo_mode": prediction.is_demo_mode,
            "model_source": prediction.model_source,
            "warning": prediction.warning_hi if lang == "hi" else prediction.warning_en
        },
        "diagnosis": diagnosis,
        "weather_context": {
            "current": weather_data.get("current") if weather_data else None,
            "active_alerts": weather_alerts
        },
        "kisan_call_centre": config.KISAN_CALL_CENTRE
    }


@app.get("/diseases")
def list_diseases(lang: str = Query("en", description="Language: 'en' or 'hi'")):
    """Returns knowledge base of all supported crop diseases."""
    all_diseases = remedy_engine.get_all_supported_diseases()
    formatted = {}
    for cls_name in all_diseases:
        formatted[cls_name] = remedy_engine.get_diagnosis(cls_name, lang=lang)
    return {"count": len(formatted), "diseases": formatted}


@app.get("/weather")
def get_weather_advisory(
    latitude: float = Query(28.6139, description="Field latitude"),
    longitude: float = Query(77.2090, description="Field longitude")
):
    """Fetches real-time agro-weather and evaluates disease outbreak susceptibility."""
    weather_data = weather_advisor.get_weather(latitude, longitude)
    disease_info = remedy_engine.get_all_supported_diseases()
    alerts = weather_advisor.evaluate_disease_risks(weather_data, disease_info)
    return {
        "weather": weather_data,
        "disease_susceptibility_alerts": alerts
    }


@app.get("/history")
def get_scan_history(limit: int = Query(50, description="Number of recent records")):
    """Retrieve recent scan history and summary metrics."""
    records = history_db.get_recent_scans(limit=limit)
    analytics = history_db.get_analytics()
    return {
        "analytics": analytics,
        "recent_scans": records
    }


@app.post("/history/clear")
def clear_scan_history():
    """Clear all records from scan history."""
    history_db.clear_history()
    return {"success": True, "message": "Scan history cleared successfully."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)

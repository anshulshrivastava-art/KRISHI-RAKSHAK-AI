"""
Krishi Rakshak AI - Remedy Recommendation Engine
Team: Codex Verse
Description: Retrieves verified agronomic remedies, cultural management, organic options,
             and chemical treatment guidance based on diagnostic predictions.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import config


class RemedyEngine:
    """Provides structured disease information and actionable remedies for farmers."""

    def __init__(self, disease_info_path: Optional[Path] = None):
        self.path = disease_info_path or config.DISEASE_INFO_PATH
        self.disease_data: Dict[str, Any] = self._load_data()

    def _load_data(self) -> Dict[str, Any]:
        """Loads disease knowledge base from JSON file."""
        if not self.path.exists():
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def get_diagnosis(self, class_name: str, lang: str = "en") -> Dict[str, Any]:
        """
        Retrieve structured diagnosis, symptoms, and remedies for a given class name.
        Defaults to English if requested language is unavailable.
        """
        lang_key = "hi" if lang == "hi" else "en"
        raw_info = self.disease_data.get(class_name)

        if not raw_info:
            # Fallback for unrecognized class
            return {
                "class_name": class_name,
                "crop_name": "Unknown Crop" if lang_key == "en" else "अज्ञात फसल",
                "disease_name": "Unrecognized Condition" if lang_key == "en" else "अज्ञात स्थिति",
                "status": "uncertain",
                "is_healthy": False,
                "pathogen": "N/A",
                "severity": "Unknown",
                "symptoms": [
                    "No verified visual profile matches this class in the database."
                    if lang_key == "en" else
                    "डेटाबेस में इस पत्ती से मेल खाने वाला सत्यापित लक्षण उपलब्ध नहीं है।"
                ],
                "causes": [],
                "preventive_measures": [],
                "treatment": {"cultural": [], "organic_biological": [], "chemical_guidance": []},
                "helpline": config.KISAN_CALL_CENTRE
            }

        # Format localized content
        crop_name = raw_info["crop_name"].get(lang_key, raw_info["crop_name"]["en"])
        disease_name = raw_info["disease_name"].get(lang_key, raw_info["disease_name"]["en"])
        is_healthy = (raw_info.get("status") == "healthy")

        symptoms = raw_info.get("symptoms", {}).get(lang_key, raw_info.get("symptoms", {}).get("en", []))
        causes = raw_info.get("causes", {}).get(lang_key, raw_info.get("causes", {}).get("en", []))
        preventive = raw_info.get("preventive_measures", {}).get(lang_key, raw_info.get("preventive_measures", {}).get("en", []))
        
        treatment_raw = raw_info.get("treatment", {})
        cultural = treatment_raw.get("cultural", {}).get(lang_key, treatment_raw.get("cultural", {}).get("en", []))
        organic = treatment_raw.get("organic_biological", {}).get(lang_key, treatment_raw.get("organic_biological", {}).get("en", []))
        chemical = treatment_raw.get("chemical_guidance", {}).get(lang_key, treatment_raw.get("chemical_guidance", {}).get("en", []))

        weather_cond = raw_info.get("weather_favorable_conditions", {})
        advisory = weather_cond.get("advisory", {}).get(lang_key, weather_cond.get("advisory", {}).get("en", ""))

        return {
            "class_name": class_name,
            "crop_name": crop_name,
            "disease_name": disease_name,
            "status": raw_info.get("status", "diseased"),
            "is_healthy": is_healthy,
            "pathogen": raw_info.get("pathogen", "N/A"),
            "severity": raw_info.get("severity", "Moderate"),
            "symptoms": symptoms,
            "causes": causes,
            "preventive_measures": preventive,
            "treatment": {
                "cultural": cultural,
                "organic_biological": organic,
                "chemical_guidance": chemical
            },
            "weather_advisory": advisory,
            "helpline": config.KISAN_CALL_CENTRE
        }

    def get_all_supported_diseases(self) -> Dict[str, Any]:
        """Return the raw disease knowledge base."""
        return self.disease_data

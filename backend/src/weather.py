"""
Krishi Rakshak AI - Weather Advisory Module
Team: Codex Verse
Description: Integrates with Open-Meteo API to fetch real-time agro-meteorological data
             and evaluates rule-based crop disease susceptibility alerts.
"""

from typing import Dict, Any, List, Optional
import requests
import config


# WMO Weather interpretation mapping
WMO_CODE_MAP = {
    0: {"en": "Clear Sky", "hi": "साफ आसमान", "icon": "☀️"},
    1: {"en": "Mainly Clear", "hi": "मुख्यतः साफ", "icon": "🌤️"},
    2: {"en": "Partly Cloudy", "hi": "आंशिक बादल", "icon": "⛅"},
    3: {"en": "Overcast", "hi": "घने बादल", "icon": "☁️"},
    45: {"en": "Foggy", "hi": "कोहरा", "icon": "🌫️"},
    48: {"en": "Depositing Rime Fog", "hi": "घना कोहरा", "icon": "🌫️"},
    51: {"en": "Light Drizzle", "hi": "हल्की बूंदाबांदी", "icon": "🌦️"},
    53: {"en": "Moderate Drizzle", "hi": "बूंदाबांदी", "icon": "🌧️"},
    55: {"en": "Dense Drizzle", "hi": "तेज बूंदाबांदी", "icon": "🌧️"},
    61: {"en": "Slight Rain", "hi": "हल्की बारिश", "icon": "🌧️"},
    63: {"en": "Moderate Rain", "hi": "मध्यम बारिश", "icon": "🌧️"},
    65: {"en": "Heavy Rain", "hi": "भारी बारिश", "icon": "⛈️"},
    80: {"en": "Rain Showers", "hi": "बारिश की बौछारें", "icon": "🌦️"},
    81: {"en": "Moderate Showers", "hi": "तेज बौछारें", "icon": "🌧️"},
    82: {"en": "Violent Showers", "hi": "मूसलाधार बारिश", "icon": "⛈️"},
    95: {"en": "Thunderstorm", "hi": "गरज-चमक के साथ बारिश", "icon": "⚡"}
}


class WeatherAdvisor:
    """Fetches agrometeorological data and computes crop pathogen risk advisories."""

    def __init__(self, timeout: float = config.WEATHER_TIMEOUT):
        self.timeout = timeout
        self.base_url = config.WEATHER_API_URL

    def get_weather(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """
        Fetch real-time weather and 3-day forecast from Open-Meteo API.
        Handles network errors gracefully.
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum",
            "timezone": "auto",
            "forecast_days": 4
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            current = data.get("current", {})
            daily = data.get("daily", {})
            wmo_code = current.get("weather_code", 0)
            condition_info = WMO_CODE_MAP.get(wmo_code, {"en": "Variable", "hi": "परिवर्तनशील", "icon": "⛅"})
            
            # Format 3-day forecast
            forecast_days = []
            dates = daily.get("time", [])
            max_temps = daily.get("temperature_2m_max", [])
            min_temps = daily.get("temperature_2m_min", [])
            precip_sums = daily.get("precipitation_sum", [])
            daily_codes = daily.get("weather_code", [])
            
            for i in range(1, min(len(dates), 4)):
                day_code = daily_codes[i] if i < len(daily_codes) else 0
                day_cond = WMO_CODE_MAP.get(day_code, {"en": "Fair", "hi": "सामान्य", "icon": "🌤️"})
                forecast_days.append({
                    "date": dates[i],
                    "max_temp": max_temps[i] if i < len(max_temps) else 25.0,
                    "min_temp": min_temps[i] if i < len(min_temps) else 15.0,
                    "precipitation": precip_sums[i] if i < len(precip_sums) else 0.0,
                    "condition_en": day_cond["en"],
                    "condition_hi": day_cond["hi"],
                    "icon": day_cond["icon"]
                })
                
            return {
                "success": True,
                "current": {
                    "temperature": current.get("temperature_2m", 25.0),
                    "humidity": current.get("relative_humidity_2m", 60.0),
                    "precipitation": current.get("precipitation", 0.0),
                    "wind_speed": current.get("wind_speed_10m", 5.0),
                    "condition_en": condition_info["en"],
                    "condition_hi": condition_info["hi"],
                    "icon": condition_info["icon"]
                },
                "forecast": forecast_days,
                "offline": False
            }
            
        except requests.RequestException as e:
            # Graceful network failure fallback
            return {
                "success": False,
                "error": str(e),
                "offline": True,
                "current": {
                    "temperature": 25.0,
                    "humidity": 65.0,
                    "precipitation": 0.0,
                    "wind_speed": 5.0,
                    "condition_en": "Unavailable (Offline)",
                    "condition_hi": "उपलब्ध नहीं (ऑफ़लाइन)",
                    "icon": "📡"
                },
                "forecast": []
            }

    def evaluate_disease_risks(
        self,
        weather_data: Dict[str, Any],
        disease_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate rule-based disease susceptibility alerts based on current
        temperature, humidity, and rainfall against crop pathogen requirements.
        """
        if not weather_data.get("current"):
            return []
            
        temp = weather_data["current"]["temperature"]
        humidity = weather_data["current"]["humidity"]
        precipitation = weather_data["current"]["precipitation"]
        
        alerts = []
        
        for class_name, info in disease_info.items():
            if info.get("status") != "diseased":
                continue
                
            cond = info.get("weather_favorable_conditions")
            if not cond:
                continue
                
            min_t = cond.get("min_temp_c", 0)
            max_t = cond.get("max_temp_c", 50)
            min_h = cond.get("min_humidity_pct", 100)
            rain_risk = cond.get("rainfall_risk", False)
            
            # Rule checking
            temp_match = min_t <= temp <= max_t
            humidity_match = humidity >= (min_h - 5.0)  # within 5% threshold
            rain_match = (precipitation > 0.1) if rain_risk else True
            
            if temp_match and humidity_match:
                risk_level = "High" if (humidity >= min_h and (not rain_risk or precipitation > 0.5)) else "Moderate"
                
                alerts.append({
                    "class_name": class_name,
                    "crop_en": info["crop_name"]["en"],
                    "crop_hi": info["crop_name"]["hi"],
                    "disease_en": info["disease_name"]["en"],
                    "disease_hi": info["disease_name"]["hi"],
                    "risk_level": risk_level,
                    "advisory_en": cond.get("advisory", {}).get("en", "Favorable weather for disease development."),
                    "advisory_hi": cond.get("advisory", {}).get("hi", "रोग पनपने के लिए मौसम अनुकूल है।"),
                    "matched_criteria": f"Temp: {temp}°C, Humidity: {humidity}%"
                })
                
        # Sort by risk level (High first)
        alerts.sort(key=lambda x: 0 if x["risk_level"] == "High" else 1)
        return alerts

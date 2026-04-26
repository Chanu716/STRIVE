"""OpenWeatherMap client with a five-minute in-process cache."""

from __future__ import annotations

import os
import time

import httpx
from dotenv import load_dotenv


load_dotenv()

CACHE_TTL = 300
_cache: dict[tuple[float, float], tuple[float, dict[str, float]]] = {}


def _parse_weather(payload: dict) -> dict[str, float]:
    """Normalize the OpenWeatherMap response into model features."""
    rain = payload.get("rain") or {}
    snow = payload.get("snow") or {}
    precipitation_mm = float(rain.get("1h", 0.0)) + float(snow.get("1h", 0.0))
    visibility_km = float(payload.get("visibility", 10000)) / 1000.0
    wind_speed_ms = float((payload.get("wind") or {}).get("speed", 0.0))
    temperature_c = float((payload.get("main") or {}).get("temp", 20.0))
    return {
        "precipitation_mm": precipitation_mm,
        "visibility_km": visibility_km,
        "wind_speed_ms": wind_speed_ms,
        "temperature_c": temperature_c,
    }


def get_weather(lat: float, lon: float) -> dict[str, float]:
    """Fetch cached weather features for a coordinate pair."""
    key = (round(lat, 2), round(lon, 2))
    now = time.time()
    cached = _cache.get(key)
    if cached and now - cached[0] < CACHE_TTL:
        return cached[1]

    api_key = os.getenv("OWM_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        data = {
            "precipitation_mm": 0.0,
            "visibility_km": 10.0,
            "wind_speed_ms": 0.0,
            "temperature_c": 20.0,
        }
        _cache[key] = (now, data)
        return data

    response = httpx.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"lat": lat, "lon": lon, "appid": api_key, "units": "metric"},
        timeout=5.0,
    )
    response.raise_for_status()
    data = _parse_weather(response.json())
    _cache[key] = (now, data)
    return data

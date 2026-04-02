"""
Dryft — Weather Module (Tomorrow.io)

Hyperlocal weather via coordinates from environment variables.
Used for keyword-triggered injection in proxy.py and morning message.
"""

import json
import os
from urllib.request import urlopen, Request
from urllib.error import URLError

# Location from environment (set WEATHER_LAT and WEATHER_LON in .env)
LATITUDE = float(os.environ.get("WEATHER_LAT", "0"))
LONGITUDE = float(os.environ.get("WEATHER_LON", "0"))

# Tomorrow.io API base
TOMORROW_API_URL = "https://api.tomorrow.io/v4/weather"

# Weather code descriptions
WEATHER_CODES = {
    0: "Unknown",
    1000: "Clear",
    1100: "Mostly Clear",
    1101: "Partly Cloudy",
    1102: "Mostly Cloudy",
    1001: "Cloudy",
    2000: "Fog",
    2100: "Light Fog",
    4000: "Drizzle",
    4001: "Rain",
    4200: "Light Rain",
    4201: "Heavy Rain",
    5000: "Snow",
    5001: "Flurries",
    5100: "Light Snow",
    5101: "Heavy Snow",
    6000: "Freezing Drizzle",
    6001: "Freezing Rain",
    6200: "Light Freezing Rain",
    6201: "Heavy Freezing Rain",
    7000: "Ice Pellets",
    7101: "Heavy Ice Pellets",
    7102: "Light Ice Pellets",
    8000: "Thunderstorm",
}

# Wind direction labels
def _wind_direction(degrees: float) -> str:
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                   "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = round(degrees / 22.5) % 16
    return directions[idx]


def _fetch_json(url: str) -> dict:
    """Fetch JSON from a URL."""
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_weather_summary(api_key: str, include_hourly: bool = False) -> str:
    """
    Returns a plain English weather summary for Red Deer County.
    Current conditions + today's forecast + frost/precip alerts if present.
    Optional hourly breakdown for morning message use.
    """
    try:
        # Realtime conditions
        realtime_url = (
            f"{TOMORROW_API_URL}/realtime"
            f"?location={LATITUDE},{LONGITUDE}"
            f"&apikey={api_key}"
            f"&units=metric"
        )
        realtime = _fetch_json(realtime_url)
        current = realtime.get("data", {}).get("values", {})

        temp = current.get("temperature", "?")
        feels = current.get("temperatureApparent", "?")
        wind_speed = current.get("windSpeed", 0)
        wind_dir = current.get("windDirection", 0)
        weather_code = current.get("weatherCode", 0)
        uv = current.get("uvIndex", 0)

        condition = WEATHER_CODES.get(weather_code, "Unknown")
        wind_label = _wind_direction(wind_dir) if wind_dir else ""

        parts = [f"Red Deer County: {temp}\u00b0C (feels like {feels}\u00b0C). {condition}."]

        if wind_speed and wind_speed > 5:
            parts.append(f"Wind {wind_label} {wind_speed:.0f} km/h.")

        if uv and uv >= 6:
            parts.append(f"UV index {uv} (high).")

        # Daily forecast
        forecast_url = (
            f"{TOMORROW_API_URL}/forecast"
            f"?location={LATITUDE},{LONGITUDE}"
            f"&apikey={api_key}"
            f"&units=metric"
            f"&timesteps=1d"
        )
        forecast = _fetch_json(forecast_url)
        daily = forecast.get("timelines", {}).get("daily", [])

        if daily:
            today = daily[0].get("values", {})
            high = today.get("temperatureMax", "?")
            low = today.get("temperatureMin", "?")
            precip_prob = today.get("precipitationProbabilityMax", 0)
            precip_type = today.get("precipitationTypeMax", 0)

            parts.append(f"High {high}\u00b0C, low {low}\u00b0C.")

            if precip_prob and precip_prob > 20:
                precip_labels = {0: "precipitation", 1: "rain", 2: "snow",
                                 3: "freezing rain", 4: "ice pellets"}
                precip_name = precip_labels.get(precip_type, "precipitation")
                parts.append(f"{precip_prob}% chance of {precip_name}.")

            # Frost alert: overnight low below 2C
            if low != "?" and float(low) < 2:
                parts.insert(0, "\u26a0\ufe0f Frost risk tonight.")

        # Hourly breakdown for morning message
        if include_hourly:
            hourly_url = (
                f"{TOMORROW_API_URL}/forecast"
                f"?location={LATITUDE},{LONGITUDE}"
                f"&apikey={api_key}"
                f"&units=metric"
                f"&timesteps=1h"
            )
            hourly_data = _fetch_json(hourly_url)
            hourly = hourly_data.get("timelines", {}).get("hourly", [])

            if hourly:
                hour_parts = []
                for h in hourly[:12]:  # next 12 hours
                    vals = h.get("values", {})
                    time_str = h.get("time", "")
                    h_temp = vals.get("temperature", "?")
                    h_code = vals.get("weatherCode", 0)
                    h_cond = WEATHER_CODES.get(h_code, "")
                    # Extract hour from ISO timestamp
                    hour_label = time_str[11:16] if len(time_str) > 16 else time_str
                    hour_parts.append(f"{hour_label}: {h_temp}\u00b0C {h_cond}")
                if hour_parts:
                    parts.append("Hourly: " + " | ".join(hour_parts))

        return " ".join(parts)

    except URLError as e:
        return f"Weather unavailable: {e}"
    except Exception as e:
        return f"Weather fetch failed: {e}"

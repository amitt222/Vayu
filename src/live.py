"""
Vayu - Live Data Module
Author: Vayu Team
Description:
    Fetches current air quality and weather for the monitored cities from the free,
    key-less Open-Meteo APIs, converts hourly concentrations into daily CPCB AQI values
    with the official averaging rules, and rebuilds the exact feature matrix used in
    training so the trained LightGBM models can produce live 24/48/72-hour forecasts.

Data sources:
    - Air quality: Open-Meteo Air Quality API (Copernicus CAMS global model, ~45 km grid).
      These are model estimates, not CPCB ground-station measurements, so absolute
      levels can differ from what nearby monitors report.
    - Weather: Open-Meteo Forecast API (same variables and units as training data).

CPCB averaging rules applied:
    - PM2.5, PM10, NO2, SO2, NH3: 24-hour mean (at least 16 valid hours).
    - CO, O3: maximum 8-hour running mean (at least 6 valid hours per window).
    - AQI is reported only when at least 3 pollutants (one of them PM2.5 or PM10) are valid.

Ozone note:
    CAMS is known to over-estimate surface ozone over South Asia. Taken at face value it
    would set the AQI on most days (e.g. 'Very Poor' in Mumbai from ozone alone), unlike
    CPCB station data, where particulates dominate. Live AQI therefore excludes O3 from the
    maximum (LIVE_AQI_EXCLUDED); O3 is still fetched and shown separately in the app.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from src.advisory import calculate_sub_index
from src.features import add_lag_features, add_rolling_features, add_temporal_features

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo variable -> dataset column. CO is converted from µg/m³ to mg/m³.
HOURLY_VARIABLES: Dict[str, str] = {
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "nitrogen_dioxide": "NO2",
    "sulphur_dioxide": "SO2",
    "ozone": "O3",
    "carbon_monoxide": "CO",
    "ammonia": "NH3",
}
AVG_24H_POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "NH3"]
MAX_8H_POLLUTANTS = ["CO", "O3"]
WEATHER_VARIABLES: Dict[str, str] = {
    "temperature_2m_mean": "temp_mean",
    "relative_humidity_2m_mean": "humidity_mean",
    "precipitation_sum": "precipitation",
    "wind_speed_10m_max": "wind_speed_max",
    "wind_direction_10m_dominant": "wind_direction",
}
# Pollutants used by the model that the live feed does not provide; filled with each
# city's recent historical median so the model sees typical values.
UNAVAILABLE_MODEL_POLLUTANTS = ["NO", "NOx", "NH3", "Benzene", "Toluene", "Xylene"]

LIVE_AQI_EXCLUDED = ("O3",)

PAST_DAYS = 45      # enough history for 14-day rolling windows + 7-day lags
FORECAST_DAYS = 4   # today + 3 days (CAMS reference forecast for comparison)


def _coordinate_params(cities: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    names = list(cities)
    return {
        "latitude": ",".join(f"{cities[c]['lat']:.4f}" for c in names),
        "longitude": ",".join(f"{cities[c]['lon']:.4f}" for c in names),
    }


def _as_list(payload: Any) -> List[Dict[str, Any]]:
    """Open-Meteo returns an object for one location and a list for several."""
    if isinstance(payload, dict):
        if payload.get("error"):
            raise RuntimeError(payload.get("reason", "Open-Meteo returned an error"))
        return [payload]
    return list(payload)


def fetch_hourly_air_quality(
    cities: Dict[str, Dict[str, float]],
    past_days: int = PAST_DAYS,
    forecast_days: int = FORECAST_DAYS,
    timeout: int = 30,
) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """
    Fetches hourly pollutant concentrations for all cities in one request.

    Returns:
        (hourly DataFrame [City, time, PM2.5, PM10, NO2, SO2, O3, CO (mg/m³), NH3],
         current local time in India floored to the hour)
    """
    params = {
        **_coordinate_params(cities),
        "hourly": ",".join(HOURLY_VARIABLES),
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "Asia/Kolkata",
    }
    response = requests.get(AIR_QUALITY_URL, params=params, timeout=timeout)
    response.raise_for_status()
    locations = _as_list(response.json())
    if len(locations) != len(cities):
        raise RuntimeError("Unexpected number of locations in the air quality response")

    frames = []
    offset_seconds = int(locations[0].get("utc_offset_seconds", 19800))
    for city, loc in zip(cities, locations):
        hourly = loc.get("hourly", {})
        frame = pd.DataFrame({"time": pd.to_datetime(hourly.get("time", []))})
        for api_name, column in HOURLY_VARIABLES.items():
            values = hourly.get(api_name)
            frame[column] = pd.to_numeric(pd.Series(values, dtype="float64"), errors="coerce") if values else np.nan
        frame["CO"] = frame["CO"] / 1000.0  # µg/m³ -> mg/m³ (CPCB unit)
        frame.insert(0, "City", city)
        frames.append(frame)

    now_local = (pd.Timestamp.now(tz="UTC").tz_localize(None)
                 + pd.Timedelta(seconds=offset_seconds)).floor("h")
    return pd.concat(frames, ignore_index=True), now_local


def fetch_daily_weather(
    cities: Dict[str, Dict[str, float]],
    past_days: int = PAST_DAYS,
    forecast_days: int = FORECAST_DAYS,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetches daily weather (same variables/units as training) for all cities."""
    params = {
        **_coordinate_params(cities),
        "daily": ",".join(WEATHER_VARIABLES),
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "Asia/Kolkata",
    }
    response = requests.get(WEATHER_URL, params=params, timeout=timeout)
    response.raise_for_status()
    locations = _as_list(response.json())
    if len(locations) != len(cities):
        raise RuntimeError("Unexpected number of locations in the weather response")

    frames = []
    for city, loc in zip(cities, locations):
        daily = loc.get("daily", {})
        frame = pd.DataFrame({"Date": pd.to_datetime(daily.get("time", []))})
        frame["City"] = city
        for api_name, column in WEATHER_VARIABLES.items():
            frame[column] = pd.to_numeric(pd.Series(daily.get(api_name, []), dtype="float64"), errors="coerce")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def cpcb_aqi_from_concentrations(
    conc: Dict[str, float], exclude: Tuple[str, ...] = LIVE_AQI_EXCLUDED
) -> Tuple[float, Optional[str]]:
    """
    Overall CPCB AQI = highest sub-index, valid only with >= 3 pollutants including a PM.
    Pollutants in `exclude` are ignored (see the ozone note in the module docstring).
    Returns (aqi or NaN, dominant pollutant or None).
    """
    pollutants = [p for p in AVG_24H_POLLUTANTS + MAX_8H_POLLUTANTS if p not in exclude]
    subs = {p: calculate_sub_index(p, conc.get(p)) for p in pollutants}
    subs = {p: v for p, v in subs.items() if v is not None}
    if len(subs) < 3 or not ({"PM2.5", "PM10"} & set(subs)):
        return float("nan"), None
    dominant = max(subs, key=subs.get)
    return float(subs[dominant]), dominant


def _window_concentrations(
    city_hourly: pd.DataFrame, rolling_8h: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> Dict[str, float]:
    """CPCB concentrations for the 24-hour window (start, end] (both hour stamps)."""
    mask = (city_hourly.index > start) & (city_hourly.index <= end)
    window = city_hourly.loc[mask]
    out: Dict[str, float] = {}
    for p in AVG_24H_POLLUTANTS:
        valid = window[p].dropna()
        out[p] = float(valid.mean()) if len(valid) >= 16 else float("nan")
    for p in MAX_8H_POLLUTANTS:
        valid = rolling_8h.loc[mask, p].dropna()
        out[p] = float(valid.max()) if len(valid) >= 16 else float("nan")
    return out


def build_daily_aqi(hourly: pd.DataFrame, now: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Converts hourly data into daily CPCB records.

    Returns:
        observed: one row per city per past calendar day, plus a final 'today' row whose
                  window is the trailing 24 hours ending at the current hour (the same
                  convention CPCB uses for its live 'current AQI').
        cams_forecast: daily AQI computed from CAMS forecast hours for tomorrow onward
                  (a physics-model reference, shown next to the ML forecast).
    """
    today = now.normalize()
    observed_rows, forecast_rows = [], []
    for city, grp in hourly.groupby("City", sort=False):
        series = grp.set_index("time").sort_index().drop(columns="City")
        rolling_8h = series[MAX_8H_POLLUTANTS].rolling(8, min_periods=6).mean()
        days = sorted(series.index.normalize().unique())
        for day in days:
            if day < today:
                start, end, kind = day - pd.Timedelta(hours=1), day + pd.Timedelta(hours=23), "observed"
            elif day == today:
                start, end, kind = now - pd.Timedelta(hours=24), now, "observed"
            else:
                start, end, kind = day - pd.Timedelta(hours=1), day + pd.Timedelta(hours=23), "forecast"
            conc = _window_concentrations(series, rolling_8h, start, end)
            aqi, dominant = cpcb_aqi_from_concentrations(conc)
            row = {"Date": day, "City": city, **conc, "AQI": aqi, "Dominant": dominant}
            (observed_rows if kind == "observed" else forecast_rows).append(row)
    return pd.DataFrame(observed_rows), pd.DataFrame(forecast_rows)


def build_live_features(
    daily_observed: pd.DataFrame,
    weather: pd.DataFrame,
    archive_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rebuilds the training feature matrix from live daily records using the same
    functions as src/features.py (calendar, lag and shifted rolling features).
    """
    daily = daily_observed.merge(weather, on=["City", "Date"], how="left")
    weather_cols = list(WEATHER_VARIABLES.values())
    daily[weather_cols] = daily.groupby("City")[weather_cols].transform(lambda s: s.ffill().bfill())

    # Fill pollutants the live feed lacks with each city's last-two-years archive median.
    recent = archive_df[archive_df["Date"] >= archive_df["Date"].max() - pd.DateOffset(years=2)]
    medians = recent.groupby("City")[UNAVAILABLE_MODEL_POLLUTANTS].median()
    global_medians = recent[UNAVAILABLE_MODEL_POLLUTANTS].median()
    for col in UNAVAILABLE_MODEL_POLLUTANTS:
        fill = daily["City"].map(medians[col]).fillna(global_medians[col])
        daily[col] = daily[col].fillna(fill) if col in daily.columns else fill

    daily = daily.sort_values(["City", "Date"]).reset_index(drop=True)
    frames = []
    for _, grp in daily.groupby("City", sort=False):
        city_df = grp.reset_index(drop=True)
        city_df = add_temporal_features(city_df)
        city_df = add_lag_features(city_df)
        city_df = add_rolling_features(city_df)
        frames.append(city_df)
    features = pd.concat(frames, ignore_index=True)
    return features.dropna(subset=["AQI"]).reset_index(drop=True)


def fetch_live_bundle(
    cities: Dict[str, Dict[str, float]],
    archive_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    One-call entry point used by the app.

    Returns a dict with:
        features: live feature matrix (one row per city per day, latest row = now)
        hourly: hourly concentrations including CAMS forecast hours
        cams_forecast: daily CAMS reference AQI for the next days
        now: current local time (IST) used for the latest window
        fetched_at: fetch timestamp (IST)
    """
    hourly, now = fetch_hourly_air_quality(cities)
    weather = fetch_daily_weather(cities)
    observed, cams_forecast = build_daily_aqi(hourly, now)
    features = build_live_features(observed, weather, archive_df)

    latest_dates = features.groupby("City")["Date"].max()
    if features.empty or (latest_dates < now.normalize()).any():
        missing = sorted(set(cities) - set(latest_dates[latest_dates >= now.normalize()].index))
        raise RuntimeError(f"Live AQI unavailable for: {', '.join(missing) or 'all cities'}")

    return {
        "features": features,
        "hourly": hourly,
        "cams_forecast": cams_forecast,
        "now": now,
        "fetched_at": now,
    }


def hourly_outlook(hourly: pd.DataFrame, city: str, now: pd.Timestamp,
                   hours_back: int = 48, hours_ahead: int = 48) -> pd.DataFrame:
    """Hourly PM2.5/PM10 for one city around the current hour, flagged past/forecast."""
    city_df = hourly[hourly["City"] == city].sort_values("time")
    window = city_df[(city_df["time"] > now - pd.Timedelta(hours=hours_back))
                     & (city_df["time"] <= now + pd.Timedelta(hours=hours_ahead))].copy()
    window["is_forecast"] = window["time"] > now
    window["PM2.5_index"] = window["PM2.5"].map(lambda v: calculate_sub_index("PM2.5", v))
    return window.reset_index(drop=True)


def best_outdoor_window(outlook: pd.DataFrame, now: pd.Timestamp, length_hours: int = 2,
                        day_start: int = 6, day_end: int = 21) -> Optional[Dict[str, Any]]:
    """
    Finds the daytime window (default 06:00–21:00) in the next 24 hours with the lowest
    mean forecast PM2.5. Returns start/end times and the mean concentration.
    """
    future = outlook[(outlook["time"] > now) & (outlook["time"] <= now + pd.Timedelta(hours=24))]
    future = future.set_index("time")["PM2.5"].dropna()
    if len(future) < length_hours:
        return None
    rolling = future.rolling(length_hours).mean().dropna()
    starts = rolling.index - pd.Timedelta(hours=length_hours - 1)
    daytime = [(s, v) for s, v in zip(starts, rolling.values)
               if s.hour >= day_start and (s + pd.Timedelta(hours=length_hours)).hour <= day_end
               and (s + pd.Timedelta(hours=length_hours)).normalize() == s.normalize()]
    if not daytime:
        return None
    start, value = min(daytime, key=lambda x: x[1])
    return {"start": start, "end": start + pd.Timedelta(hours=length_hours), "pm25": float(value)}

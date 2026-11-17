"""
Vayu - Streamlit App Helper Utilities
Author: Vayu Team
Description:
    Shared data loading, model inference, multi-horizon forecasting, calibrated
    prediction intervals, hold-out evaluation, SHAP explanations, what-if scenarios
    and analytical helpers for the Streamlit web application.

    Everything in this module is framework-agnostic (no Streamlit imports) so it
    can be unit-tested and reused by scripts; caching is applied in app.py.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.advisory import get_cpcb_category

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Must match the chronological split in src/train.py (test set = Date > 2019-12-31).
TEST_START_DATE = "2020-01-01"
HORIZONS: List[int] = [1, 2, 3]
AQI_MIN_CLIP, AQI_MAX_CLIP = 10.0, 500.0

# CPCB category bin edges, consistent with get_cpcb_category (<=50 Good, <=100 Satisfactory, ...).
CPCB_BINS = [-np.inf, 50, 100, 200, 300, 400, np.inf]
CPCB_LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

# City Geographic Coordinates for Mapping
CITY_LOCATIONS: Dict[str, Dict[str, float]] = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639},
    "Chennai": {"lat": 13.0827, "lon": 80.2707},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714},
    "Jaipur": {"lat": 26.9124, "lon": 75.7873},
}

# Nationwide COVID-19 lockdown (Phases 1-4) used for the lockdown impact analysis.
LOCKDOWN_START, LOCKDOWN_END = "2020-03-25", "2020-05-31"


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------
def load_dataset() -> pd.DataFrame:
    """Loads the preprocessed historical air quality dataset."""
    path = DATA_DIR / "cleaned_city_day.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Cleaned dataset not found at {path}. Run src.preprocessing first.")
    return pd.read_csv(path, parse_dates=["Date"])


def load_features_dataset() -> pd.DataFrame:
    """Loads the engineered feature matrix with weather and lags."""
    path = DATA_DIR / "features_city_day.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Feature dataset not found at {path}. Run src.features first.")
    return pd.read_csv(path, parse_dates=["Date"])


def load_models_and_columns() -> Tuple[Any, Dict[str, Any], List[str], pd.DataFrame]:
    """
    Loads trained ML models, multi-horizon models, feature columns, and benchmark metrics.

    Returns:
        Tuple: (best_model, multi_horizon_models_dict, feature_columns_list, metrics_df)
    """
    best_model_path = MODELS_DIR / "best_model.joblib"
    multi_models_path = MODELS_DIR / "multi_horizon_models.joblib"
    cols_path = MODELS_DIR / "feature_columns.json"
    metrics_path = MODELS_DIR / "metrics_comparison.csv"

    if not best_model_path.is_file():
        raise FileNotFoundError(f"Model not found at {best_model_path}. Run src.train first.")

    best_model = joblib.load(best_model_path)
    multi_models = joblib.load(multi_models_path) if multi_models_path.is_file() else {}

    with open(cols_path, "r") as f:
        feature_cols: List[str] = json.load(f)

    metrics_df = pd.read_csv(metrics_path) if metrics_path.is_file() else pd.DataFrame()
    return best_model, multi_models, feature_cols, metrics_df


# -----------------------------------------------------------------------------
# Encoding & inference
# -----------------------------------------------------------------------------
def encode_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    One-hot encodes City exactly like training and guarantees every feature column
    exists (missing dummy columns are filled with 0). Returns a new DataFrame.
    """
    base = df.reset_index(drop=True)
    dummies = pd.get_dummies(base["City"], prefix="City", dtype=int)
    encoded = pd.concat([base, dummies], axis=1)
    missing = [c for c in feature_cols if c not in encoded.columns]
    for col in missing:
        encoded[col] = 0
    return encoded


def to_model_matrix(encoded: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    """Returns the float feature matrix in the exact training column order."""
    return encoded[feature_cols].to_numpy(dtype=float)


def clip_aqi(values: Any) -> Any:
    """Clips predictions to the valid CPCB reporting range used in production."""
    return np.clip(values, AQI_MIN_CLIP, AQI_MAX_CLIP)


def categorize_aqi(values: pd.Series) -> pd.Series:
    """Vectorised CPCB category labels for a Series of AQI values."""
    return pd.cut(values, bins=CPCB_BINS, labels=CPCB_LABELS, right=True).astype(str)


def predict_horizons(
    rows: pd.DataFrame,
    multi_models: Dict[str, Any],
    feature_cols: List[str],
) -> Dict[int, np.ndarray]:
    """
    Predicts AQI at every horizon for the given raw feature rows.
    Falls back to persistence (today's AQI) if a horizon model is missing.
    """
    X = to_model_matrix(encode_features(rows, feature_cols), feature_cols)
    preds: Dict[int, np.ndarray] = {}
    for step in HORIZONS:
        model = multi_models.get(f"target_aqi_{step}d")
        raw = rows["AQI"].to_numpy(dtype=float) if model is None else model.predict(X)
        preds[step] = clip_aqi(np.asarray(raw, dtype=float))
    return preds


def get_city_history(features_df: pd.DataFrame, city: str) -> pd.DataFrame:
    """All feature rows for a city, sorted by date."""
    return features_df[features_df["City"] == city].sort_values("Date").reset_index(drop=True)


def get_latest_city_status(features_df: pd.DataFrame, city: str) -> Dict[str, Any]:
    """
    Retrieves the most recent recorded atmospheric state for a given city,
    plus the previous day's values for day-over-day deltas.
    """
    city_df = get_city_history(features_df, city)
    if city_df.empty:
        raise ValueError(f"No records found for city: {city}")

    latest_row = city_df.iloc[-1]
    prev_row = city_df.iloc[-2] if len(city_df) > 1 else latest_row
    aqi_val = float(latest_row["AQI"])
    cat_info = get_cpcb_category(aqi_val)

    def _val(row: pd.Series, col: str, default: float) -> float:
        value = row.get(col, default)
        return round(float(default if pd.isna(value) else value), 1)

    return {
        "city": city,
        "date": str(latest_row["Date"].date()),
        "aqi": round(aqi_val, 1),
        "prev_aqi": round(float(prev_row["AQI"]), 1),
        "category": cat_info["category"],
        "color": cat_info["color"],
        "description": cat_info["description"],
        "pm25": _val(latest_row, "PM2.5", 0.0),
        "pm10": _val(latest_row, "PM10", 0.0),
        "no2": _val(latest_row, "NO2", 0.0),
        "temp": _val(latest_row, "temp_mean", 25.0),
        "humidity": _val(latest_row, "humidity_mean", 50.0),
        "precipitation": _val(latest_row, "precipitation", 0.0),
        "wind_speed": _val(latest_row, "wind_speed_max", 10.0),
        "prev_pm25": _val(prev_row, "PM2.5", 0.0),
        "prev_pm10": _val(prev_row, "PM10", 0.0),
        "prev_temp": _val(prev_row, "temp_mean", 25.0),
        "prev_wind_speed": _val(prev_row, "wind_speed_max", 10.0),
        "latest_row": latest_row,
    }


def generate_city_forecasts(
    features_df: pd.DataFrame,
    city: str,
    multi_models: Dict[str, Any],
    feature_cols: List[str],
    intervals: Optional[Dict[int, Tuple[float, float]]] = None,
) -> List[Dict[str, Any]]:
    """
    Produces 24-hour, 48-hour, and 72-hour AQI forecasts using trained multi-step models.

    Args:
        features_df: Features DataFrame.
        city: Target city.
        multi_models: Dictionary of trained models for 1d, 2d, 3d.
        feature_cols: Ordered feature names.
        intervals: Optional {horizon: (low_offset, high_offset)} empirical residual
            quantiles (see compute_forecast_intervals). When omitted, a fixed
            heuristic margin is used.

    Returns:
        3 forecast predictions with dates, predicted AQI, category, and interval bounds.
    """
    latest_row = get_city_history(features_df, city).iloc[-1:]
    preds = predict_horizons(latest_row, multi_models, feature_cols)
    latest_date = pd.to_datetime(latest_row["Date"].values[0])

    forecasts: List[Dict[str, Any]] = []
    for step in HORIZONS:
        pred_aqi = float(preds[step][0])
        cat_info = get_cpcb_category(pred_aqi)

        if intervals and step in intervals:
            low_off, high_off = intervals[step]
        else:
            margin = 15.0 + (step - 1) * 6.0
            low_off, high_off = -margin, margin

        forecasts.append({
            "horizon_hours": step * 24,
            "horizon_days": step,
            "forecast_date": str((latest_date + pd.Timedelta(days=step)).date()),
            "predicted_aqi": round(pred_aqi, 1),
            "category": cat_info["category"],
            "color": cat_info["color"],
            "lower_bound": round(max(0.0, pred_aqi + low_off), 1),
            "upper_bound": round(min(500.0, pred_aqi + high_off), 1),
        })
    return forecasts


# -----------------------------------------------------------------------------
# Hold-out evaluation & calibrated intervals
# -----------------------------------------------------------------------------
def build_test_predictions(
    features_df: pd.DataFrame,
    multi_models: Dict[str, Any],
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Runs the production multi-horizon models on the chronological hold-out period
    (2020) — data never seen during training or early stopping.

    Returns a long DataFrame: Date, City, horizon, target_date, actual, predicted,
    error, actual_category, predicted_category.
    """
    test = features_df[features_df["Date"] >= TEST_START_DATE].reset_index(drop=True)
    if test.empty:
        return pd.DataFrame()

    preds = predict_horizons(test, multi_models, feature_cols)
    frames = []
    for step in HORIZONS:
        frame = pd.DataFrame({
            "Date": test["Date"],
            "City": test["City"],
            "horizon": step,
            "target_date": test["Date"] + pd.Timedelta(days=step),
            "actual": test[f"target_aqi_{step}d"].astype(float),
            "predicted": preds[step],
        })
        frames.append(frame.dropna(subset=["actual"]))

    out = pd.concat(frames, ignore_index=True)
    out["error"] = out["actual"] - out["predicted"]
    out["actual_category"] = categorize_aqi(out["actual"])
    out["predicted_category"] = categorize_aqi(out["predicted"])
    return out


def compute_forecast_intervals(
    test_preds: pd.DataFrame,
    coverage: float = 0.90,
) -> Dict[int, Tuple[float, float]]:
    """
    Empirical (split-conformal style) prediction intervals: the lower/upper
    quantiles of hold-out residuals (actual - predicted) per horizon.
    A forecast's interval is [pred + low, pred + high].
    """
    alpha = (1.0 - coverage) / 2.0
    intervals: Dict[int, Tuple[float, float]] = {}
    for step in HORIZONS:
        errors = test_preds.loc[test_preds["horizon"] == step, "error"] if not test_preds.empty else pd.Series(dtype=float)
        if len(errors) < 30:
            continue
        intervals[step] = (float(errors.quantile(alpha)), float(errors.quantile(1.0 - alpha)))
    return intervals


def _regression_scores(actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
    err = actual - predicted
    ss_res = float((err ** 2).sum())
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    return {
        "MAE": float(err.abs().mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "R2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def summarize_horizon_metrics(test_preds: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per horizon: MAE, RMSE, R², exact-category hit rate, within-one-category rate."""
    rows = []
    order = {label: i for i, label in enumerate(CPCB_LABELS)}
    for step, grp in test_preds.groupby("horizon"):
        scores = _regression_scores(grp["actual"], grp["predicted"])
        cat_gap = (grp["actual_category"].map(order) - grp["predicted_category"].map(order)).abs()
        rows.append({
            "Horizon": f"+{int(step) * 24}h",
            "horizon": int(step),
            **scores,
            "Category Accuracy (%)": float((cat_gap == 0).mean() * 100),
            "Within ±1 Category (%)": float((cat_gap <= 1).mean() * 100),
            "Samples": int(len(grp)),
        })
    return pd.DataFrame(rows)


def summarize_city_metrics(test_preds: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Per-city accuracy for one horizon, sorted best (lowest MAE) first."""
    subset = test_preds[test_preds["horizon"] == horizon]
    rows = []
    for city, grp in subset.groupby("City"):
        scores = _regression_scores(grp["actual"], grp["predicted"])
        rows.append({
            "City": city,
            **scores,
            "Category Accuracy (%)": float((grp["actual_category"] == grp["predicted_category"]).mean() * 100),
            "Mean AQI": float(grp["actual"].mean()),
            "Samples": int(len(grp)),
        })
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


# -----------------------------------------------------------------------------
# All-city snapshot (map, rankings, alert centre)
# -----------------------------------------------------------------------------
def build_city_snapshot(
    features_df: pd.DataFrame,
    multi_models: Dict[str, Any],
    feature_cols: List[str],
    intervals: Optional[Dict[int, Tuple[float, float]]] = None,
    trend_days: int = 30,
) -> pd.DataFrame:
    """
    One row per city with latest observation, 24/48/72h forecasts, interval bounds
    and a recent AQI trend list (for sparkline columns).
    """
    rows = []
    for city in sorted(features_df["City"].unique()):
        history = get_city_history(features_df, city)
        if history.empty:
            continue
        status = get_latest_city_status(features_df, city)
        forecasts = generate_city_forecasts(features_df, city, multi_models, feature_cols, intervals)
        row = {
            "City": city,
            "Date": status["date"],
            "AQI": status["aqi"],
            "Category": status["category"],
            "Color": status["color"],
            "PM2.5": status["pm25"],
            "PM10": status["pm10"],
            "Temp": status["temp"],
            "Wind": status["wind_speed"],
            "Trend": history["AQI"].tail(trend_days).round(0).tolist(),
        }
        for fc in forecasts:
            h = fc["horizon_hours"]
            row[f"F{h}"] = fc["predicted_aqi"]
            row[f"F{h}_cat"] = fc["category"]
            row[f"F{h}_color"] = fc["color"]
            row[f"F{h}_hi"] = fc["upper_bound"]
            row[f"F{h}_date"] = fc["forecast_date"]
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Explainability
# -----------------------------------------------------------------------------
_LAG_PATTERN = re.compile(r"^(.+)_lag_(\d+)$")
_ROLL_PATTERN = re.compile(r"^(.+)_rolling_(mean|std|min|max)_(\d+)$")

_FRIENDLY_EXACT = {
    "AQI": "Today's AQI",
    "temp_mean": "Temperature (°C)",
    "humidity_mean": "Humidity (%)",
    "precipitation": "Rainfall (mm)",
    "wind_speed_max": "Max wind speed (km/h)",
    "wind_direction": "Wind direction (°)",
    "day_of_week": "Day of week",
    "is_weekend": "Weekend",
    "day_of_month": "Day of month",
    "month": "Month",
    "quarter": "Quarter",
    "day_of_year": "Day of year",
    "sin_month": "Seasonal cycle (sin)",
    "cos_month": "Seasonal cycle (cos)",
    "sin_day_of_week": "Weekly cycle (sin)",
    "cos_day_of_week": "Weekly cycle (cos)",
    "season": "Season",
    "AQI_diff_1_2": "AQI change (yesterday vs day before)",
    "PM_ratio_lag_1": "PM2.5/PM10 ratio (yesterday)",
}


def friendly_feature_name(name: str) -> str:
    """Human-readable label for an engineered feature name."""
    if name in _FRIENDLY_EXACT:
        return _FRIENDLY_EXACT[name]
    if name.startswith("City_"):
        return f"City = {name[5:]}"
    lag = _LAG_PATTERN.match(name)
    if lag:
        base, days = lag.group(1), int(lag.group(2))
        return f"{base} {days} day{'s' if days > 1 else ''} ago"
    roll = _ROLL_PATTERN.match(name)
    if roll:
        base, stat, days = roll.groups()
        stat_label = {"mean": "avg", "std": "volatility", "min": "min", "max": "max"}[stat]
        return f"{days}-day {stat_label} {base}"
    if name in {"PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"}:
        return f"Today's {name}"
    return name


def explain_single_prediction(
    explainer: Any,
    row: pd.DataFrame,
    feature_cols: List[str],
) -> Dict[str, Any]:
    """
    TreeSHAP attribution for one raw feature row.

    Returns base value, model output, and a DataFrame of every feature's value and
    SHAP contribution sorted by absolute impact.
    """
    X = to_model_matrix(encode_features(row, feature_cols), feature_cols)
    explanation = explainer(X)
    values = np.asarray(explanation.values)[0]
    base = float(np.ravel(explanation.base_values)[0])

    contrib = pd.DataFrame({
        "feature": feature_cols,
        "label": [friendly_feature_name(c) for c in feature_cols],
        "value": X[0],
        "shap": values,
    })
    contrib["abs_shap"] = contrib["shap"].abs()
    contrib = contrib.sort_values("abs_shap", ascending=False).reset_index(drop=True)
    return {"base_value": base, "prediction": base + float(values.sum()), "contributions": contrib}


# -----------------------------------------------------------------------------
# What-if scenarios
# -----------------------------------------------------------------------------
def scenario_forecast(
    base_row: pd.DataFrame,
    overrides: Dict[str, float],
    multi_models: Dict[str, Any],
    feature_cols: List[str],
) -> Dict[int, float]:
    """
    Re-runs the horizon models after overriding today's feature values
    (e.g. {'PM2.5': 80, 'wind_speed_max': 20}). Lags and rolling statistics are
    left unchanged, so this is a model sensitivity analysis, not a causal estimate.
    """
    row = base_row.copy()
    for col, value in overrides.items():
        if col in row.columns:
            row[col] = float(value)
    preds = predict_horizons(row, multi_models, feature_cols)
    return {step: float(preds[step][0]) for step in HORIZONS}


# -----------------------------------------------------------------------------
# Analytics
# -----------------------------------------------------------------------------
def lockdown_impact(raw_df: pd.DataFrame, cities: List[str]) -> pd.DataFrame:
    """
    Compares mean pollutant levels during the 2020 nationwide lockdown window
    with the same calendar window in 2019, per city.
    """
    start, end = pd.Timestamp(LOCKDOWN_START), pd.Timestamp(LOCKDOWN_END)
    prev_start, prev_end = start - pd.DateOffset(years=1), end - pd.DateOffset(years=1)
    rows = []
    for city in cities:
        city_df = raw_df[raw_df["City"] == city]
        during = city_df[(city_df["Date"] >= start) & (city_df["Date"] <= end)]
        before = city_df[(city_df["Date"] >= prev_start) & (city_df["Date"] <= prev_end)]
        if during.empty or before.empty:
            continue
        for metric in ["AQI", "PM2.5", "NO2"]:
            prev_mean, curr_mean = before[metric].mean(), during[metric].mean()
            if pd.isna(prev_mean) or pd.isna(curr_mean) or prev_mean == 0:
                continue
            rows.append({
                "City": city,
                "Metric": metric,
                "2019": round(float(prev_mean), 1),
                "2020 Lockdown": round(float(curr_mean), 1),
                "Change (%)": round(float((curr_mean - prev_mean) / prev_mean * 100), 1),
            })
    return pd.DataFrame(rows)


def city_summary_stats(raw_df: pd.DataFrame, cities: List[str]) -> pd.DataFrame:
    """Long-run summary per city: mean AQI, share of clean/unhealthy days, worst month."""
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    rows = []
    for city in cities:
        city_df = raw_df[raw_df["City"] == city].dropna(subset=["AQI"])
        if city_df.empty:
            continue
        monthly = city_df.groupby(city_df["Date"].dt.month)["AQI"].mean()
        rows.append({
            "City": city,
            "Mean AQI": round(float(city_df["AQI"].mean()), 1),
            "Good/Satisfactory Days (%)": round(float((city_df["AQI"] <= 100).mean() * 100), 1),
            "Poor or Worse Days (%)": round(float((city_df["AQI"] > 200).mean() * 100), 1),
            "Worst Month": months[int(monthly.idxmax()) - 1],
            "Cleanest Month": months[int(monthly.idxmin()) - 1],
            "Records": int(len(city_df)),
            "From": str(city_df["Date"].min().date()),
            "To": str(city_df["Date"].max().date()),
        })
    return pd.DataFrame(rows)


def text_color_for(hex_color: str) -> str:
    """Returns a dark or white text color with readable contrast on `hex_color`."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return "#FFFFFF"
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    def _lin(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    luminance = 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)
    # Contrast with white vs near-black (#111827, luminance ≈ 0.0122)
    contrast_white = 1.05 / (luminance + 0.05)
    contrast_dark = (luminance + 0.05) / (0.0122 + 0.05)
    return "#111827" if contrast_dark >= contrast_white else "#FFFFFF"

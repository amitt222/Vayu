"""
Vayu - Streamlit Web Application
Air Quality Forecasting & Health Alert Platform for Indian Cities

Run with:  streamlit run app/app.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = APP_DIR / "assets"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def _load_app_utils():
    """
    Loads app/utils.py by file path under a unique module name.

    `streamlit run app/app.py` puts app/ first on sys.path, so `import app` can
    resolve to this very script instead of the package (re-executing the whole app),
    and a bare `import utils` can collide with third-party packages.
    """
    name = "vayu_app_utils"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, APP_DIR / "utils.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


U = _load_app_utils()

from src import live as L
from src.advisory import (
    CPCB_SCALE,
    WHO_24H_GUIDELINES,
    calculate_outdoor_safety_score,
    calculate_sub_index,
    get_cpcb_category,
    get_health_advisory,
    get_pollutant_sub_indices,
    normalize_user_group,
    pm25_cigarette_equivalent,
)

# -----------------------------------------------------------------------------
# Page configuration & global styles
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Vayu · Air Quality Intelligence",
    page_icon=str(ASSETS_DIR / "icon.svg"),
    layout="wide",
    initial_sidebar_state="auto",
)

st.html("""
<style>
:root { --as-border: #E4E9F1; --as-shadow: 0 1px 2px rgba(16,24,40,.04), 0 6px 18px rgba(16,24,40,.06); }
[data-testid="stMainBlockContainer"] { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1400px; }

/* Header banner */
.as-banner { position: relative; overflow: hidden; border-radius: 22px; padding: 26px 30px 22px; color: #fff;
  background: radial-gradient(circle at 88% 12%, rgba(255,255,255,.20) 0, rgba(255,255,255,0) 30%),
              radial-gradient(circle at 72% 130%, rgba(6,182,212,.55) 0, rgba(6,182,212,0) 45%),
              linear-gradient(120deg, #0B1B4D 0%, #1D4ED8 58%, #0891B2 100%);
  box-shadow: 0 12px 30px rgba(29,78,216,.22); }
.as-banner-art { position: absolute; right: 26px; top: 50%; transform: translateY(-50%); opacity: .13; width: 210px; height: 210px; }
.as-banner-row { display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; position: relative; }
.as-eyebrow-w { font-size: .72rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; opacity: .8; }
.as-title-w { font-size: 2.15rem; font-weight: 800; line-height: 1.15; margin: .35rem 0 .3rem 0; position: relative; }
.as-sub-w { opacity: .86; margin: 0 0 .85rem 0; max-width: 860px; font-size: 1rem; position: relative; }
.as-banner .as-chip { background: rgba(255,255,255,.14); border-color: rgba(255,255,255,.28); color: #fff; }
.as-status { display: inline-flex; align-items: center; gap: 8px; font-size: .72rem; font-weight: 700; letter-spacing: .06em;
  padding: 5px 12px; border-radius: 999px; background: rgba(16,185,129,.22); border: 1px solid rgba(110,231,183,.55); }
.as-status.archive { background: rgba(255,255,255,.12); border-color: rgba(255,255,255,.32); }
.as-pulse { width: 8px; height: 8px; border-radius: 50%; background: #34D399; animation: as-pulse 1.8s infinite; }
@keyframes as-pulse { 0% { box-shadow: 0 0 0 0 rgba(52,211,153,.75); } 70% { box-shadow: 0 0 0 9px rgba(52,211,153,0); }
                      100% { box-shadow: 0 0 0 0 rgba(52,211,153,0); } }

/* Generic elements */
.as-context { display: flex; flex-wrap: wrap; gap: 8px; margin-top: .35rem; position: relative; }
.as-chip { display: inline-flex; align-items: center; gap: 6px; font-size: .78rem; font-weight: 500;
           padding: 3px 11px; border-radius: 999px; border: 1px solid var(--as-border); background: #fff; }
.as-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.as-pill { display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 700; font-size: .78rem; }
.as-card { border: 1px solid var(--as-border); border-radius: 16px; padding: 18px 20px; background: #fff;
           box-shadow: var(--as-shadow); height: 100%; }
.as-hero { border-radius: 20px; padding: 22px 24px; border: 1px solid var(--as-border); height: 100%; box-shadow: var(--as-shadow); }
.as-hero .as-num { font-size: 4.4rem; font-weight: 800; line-height: 1; letter-spacing: -.02em; }
.as-unit { font-size: 1rem; font-weight: 600; opacity: .65; margin-left: 6px; }
.as-label { font-size: .76rem; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; opacity: .65; }
.as-muted { opacity: .72; font-size: .85rem; }
.as-fc { border-radius: 16px; padding: 16px 18px; border: 1px solid var(--as-border); background: #fff;
         box-shadow: var(--as-shadow); height: 100%; }
.as-big { font-size: 2.3rem; font-weight: 800; line-height: 1.1; margin: 6px 0 8px 0; }
.as-row { display: flex; justify-content: space-between; align-items: center; gap: 4px 8px; flex-wrap: wrap; }
.as-bar { height: 6px; border-radius: 999px; background: rgba(148,163,184,.25); overflow: hidden; margin-top: 12px; }
.as-bar > span { display: block; height: 100%; border-radius: 999px; }
.as-tip { border-left: 4px solid #2563EB; padding: 10px 14px; border-radius: 8px; background: rgba(37,99,235,.07); }
.as-kv { display: grid; grid-template-columns: auto 1fr; gap: 8px 14px; font-size: .9rem; }
.as-ref { margin-top: 8px; padding-top: 8px; border-top: 1px dashed var(--as-border); font-size: .8rem; opacity: .75; }

/* City pulse strip */
.as-pulse-grid { display: grid; grid-template-columns: repeat(8, minmax(0, 1fr)); gap: 10px; }
@media (max-width: 1100px) { .as-pulse-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); } }
.as-city { background: #fff; border: 1px solid var(--as-border); border-radius: 14px; padding: 10px 12px 9px;
           box-shadow: var(--as-shadow); }
.as-city.sel { outline: 2px solid #2563EB; outline-offset: 1px; }
.as-city .n { font-size: .78rem; font-weight: 600; opacity: .7; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.as-city .v { font-size: 1.45rem; font-weight: 800; line-height: 1.2; }
.as-city .c { font-size: .7rem; font-weight: 600; }
.as-city .b { height: 4px; border-radius: 999px; margin-top: 6px; }

/* Sidebar now-card */
.as-side-now { border: 1px solid var(--as-border); border-radius: 14px; padding: 12px 14px; background: #fff; }

/* White cards for Streamlit containers & metrics */
[class*="st-key-card_"] { background: #fff; box-shadow: var(--as-shadow); border-color: var(--as-border) !important; }
[data-testid="stMetric"] { background: #fff; }
.as-footer { text-align: center; font-size: .78rem; opacity: .6; padding: 18px 0 4px 0; }
</style>
""")

PROFILES: Dict[str, str] = {
    "General Public": "🧑",
    "Asthma Patients": "🫁",
    "Elderly & Children": "👵",
    "Outdoor Workers": "👷",
}
HORIZON_LABELS = {1: "+24 hours", 2: "+48 hours", 3: "+72 hours"}
FORECAST_COLOR = "#7C3AED"
OBSERVED_COLOR = "#2563EB"
BANNER_ART = (
    '<svg class="as-banner-art" viewBox="0 0 48 48" fill="none" stroke="#fff" stroke-width="2.6" '
    'stroke-linecap="round"><path d="M8 17h20a6 6 0 1 0-6-6"/><path d="M8 25h27a6 6 0 1 1-6 6"/>'
    '<path d="M8 33h13"/></svg>'
)


# -----------------------------------------------------------------------------
# Cached data & model access
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading air quality records…")
def get_data():
    return U.load_dataset(), U.load_features_dataset()


@st.cache_resource(show_spinner="Loading forecasting models…")
def get_models():
    return U.load_models_and_columns()


@st.cache_data(ttl=1800, show_spinner="Fetching live air quality for 8 cities…")
def get_live_bundle() -> Dict[str, Any]:
    """Live data refreshed at most every 30 minutes. Errors are returned, not raised."""
    raw, _ = get_data()
    try:
        bundle = L.fetch_live_bundle(U.CITY_LOCATIONS, raw)
        bundle["error"] = None
        return bundle
    except Exception as exc:  # network down, API change, incomplete data …
        return {"error": f"{type(exc).__name__}: {exc}"}


def features_for(mode: str) -> pd.DataFrame:
    if mode == "live":
        return get_live_bundle()["features"]
    return get_data()[1]


@st.cache_data(show_spinner=False)
def get_test_predictions() -> pd.DataFrame:
    _, features = get_data()
    _, multi, cols, _ = get_models()
    return U.build_test_predictions(features, multi, cols)


@st.cache_data(show_spinner=False)
def get_intervals():
    return U.compute_forecast_intervals(get_test_predictions(), coverage=0.90)


@st.cache_data(show_spinner=False, max_entries=128)
def get_city_forecasts(city: str, mode: str, stamp: str) -> List[Dict[str, Any]]:
    _, multi, cols, _ = get_models()
    return U.generate_city_forecasts(features_for(mode), city, multi, cols, get_intervals())


@st.cache_data(show_spinner=False, max_entries=16)
def get_snapshot(mode: str, stamp: str) -> pd.DataFrame:
    _, multi, cols, _ = get_models()
    return U.build_city_snapshot(features_for(mode), multi, cols, get_intervals())


@st.cache_resource(show_spinner=False)
def get_explainer(horizon: int):
    import shap  # heavy import: only loaded when the explainability page is used

    _, multi, _, _ = get_models()
    return shap.TreeExplainer(multi[f"target_aqi_{horizon}d"])


@st.cache_data(show_spinner="Computing SHAP attributions…", max_entries=128)
def explain_city(city: str, horizon: int, mode: str, stamp: str) -> Dict[str, Any]:
    _, _, cols, _ = get_models()
    row = U.get_city_history(features_for(mode), city).iloc[-1:]
    return U.explain_single_prediction(get_explainer(horizon), row, cols)


@st.cache_data(show_spinner=False)
def get_dominant_pollutant_share() -> pd.DataFrame:
    """Share of days on which each pollutant has the highest CPCB sub-index, per city."""
    from src.advisory import CPCB_BREAKPOINTS

    raw, _ = get_data()
    pollutants = [p for p in CPCB_BREAKPOINTS if p in raw.columns]
    sub = pd.DataFrame(index=raw.index)
    for p in pollutants:
        bands = CPCB_BREAKPOINTS[p]
        conc = raw[p].astype(float)
        xp = [bands[0][0]] + [b[1] for b in bands]
        fp = [bands[0][2]] + [b[3] for b in bands]
        values = np.interp(conc.clip(lower=0).fillna(0), xp, fp)
        sub[p] = np.where(conc.notna(), values, np.nan)
    valid = sub.notna().any(axis=1)
    dominant = sub[valid].idxmax(axis=1)
    out = (
        pd.DataFrame({"City": raw.loc[valid, "City"], "Pollutant": dominant})
        .groupby(["City", "Pollutant"]).size().rename("Days").reset_index()
    )
    out["Share (%)"] = out["Days"] / out.groupby("City")["Days"].transform("sum") * 100
    return out


try:
    raw_df, archive_features = get_data()
    best_model, multi_models, feature_cols, metrics_df = get_models()
except Exception as exc:  # pragma: no cover - surfaced in the UI
    st.error(f"**Vayu could not start:** {exc}\n\nRun the pipeline first: "
             "`python -m src.preprocessing`, `python -m src.features`, `python -m src.train`.")
    st.stop()

CITIES = sorted(archive_features["City"].unique().tolist())
ARCHIVE_END = archive_features["Date"].max()

# Active data context, resolved once per run (see resolve_context below).
CTX: Dict[str, Any] = {}


# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------
def fmt_date(value: Any, pattern: str = "%d %b %Y") -> str:
    return pd.to_datetime(value).strftime(pattern)


def pill(text: str, color: str) -> str:
    return f'<span class="as-pill" style="background:{color};color:{U.text_color_for(color)};">{text}</span>'


def is_live() -> bool:
    return CTX.get("mode") == "live"


def data_label() -> str:
    if is_live():
        return f"Live · updated {CTX['now']:%H:%M} IST, {CTX['now']:%d %b}"
    return f"Archive · latest record {fmt_date(ARCHIVE_END)}"


def card(key: str):
    """A white, bordered card container (styled via its st-key-card_* class)."""
    return st.container(border=True, key=f"card_{key}")


def page_header(eyebrow: str, title: str, subtitle: str, show_context: bool = True,
                archive_only: bool = False) -> None:
    if archive_only:
        status = '<span class="as-status archive">🗄️ HISTORICAL ARCHIVE · 2015–2020</span>'
    elif is_live():
        status = f'<span class="as-status"><span class="as-pulse"></span>LIVE · {CTX["now"]:%H:%M} IST</span>'
    else:
        status = f'<span class="as-status archive">🗄️ ARCHIVE · {fmt_date(ARCHIVE_END)}</span>'
    chips = ""
    if show_context:
        city, profile = st.session_state["city"], st.session_state["profile"]
        chips = (
            '<div class="as-context">'
            f'<span class="as-chip">📍 {city}</span>'
            f'<span class="as-chip">{PROFILES[profile]} {profile}</span>'
            f'<span class="as-chip">🗓️ {data_label()}</span>'
            "</div>"
        )
    st.html(
        f'<div class="as-banner">{BANNER_ART}'
        f'<div class="as-banner-row"><span class="as-eyebrow-w">{eyebrow}</span>{status}</div>'
        f'<div class="as-title-w">{title}</div><p class="as-sub-w">{subtitle}</p>{chips}</div>'
    )
    if CTX.get("error") and not archive_only:
        c1, c2 = st.columns([5, 1], vertical_alignment="center")
        c1.warning("Live data is unavailable right now, so the app is showing the 2015–2020 archive. "
                   f"Details: {CTX['error']}", icon="📡")
        if c2.button("Retry live", icon=":material/refresh:", width="stretch", key=f"retry_{title}"):
            get_live_bundle.clear()
            st.rerun()


def cpcb_legend() -> None:
    chips = "".join(
        f'<span class="as-chip"><span class="as-dot" style="background:{m["color"]}"></span>'
        f'{name} <span style="opacity:.6">{m["min"]}–{m["max"]}</span></span>'
        for name, m in CPCB_SCALE.items()
    )
    st.html(f'<div class="as-context">{chips}</div>')


def cpcb_colorscale() -> List[List[Any]]:
    """Stepped Plotly colorscale for z in [0, 500] that follows CPCB bands exactly."""
    edges = [0, 50, 100, 200, 300, 400, 500]
    colors = [m["color"] for m in CPCB_SCALE.values()]
    scale: List[List[Any]] = []
    for i, color in enumerate(colors):
        scale += [[edges[i] / 500, color], [edges[i + 1] / 500, color]]
    return scale


CPCB_COLORBAR = dict(
    title="AQI", tickvals=[25, 75, 150, 250, 350, 450],
    ticktext=list(CPCB_SCALE.keys()), thickness=12, len=0.9,
)


def add_cpcb_bands(fig: go.Figure, y_max: float) -> None:
    edges = [0, 50, 100, 200, 300, 400, 500]
    for i, (name, meta) in enumerate(CPCB_SCALE.items()):
        lo, hi = edges[i], min(edges[i + 1], y_max)
        if lo >= y_max:
            break
        fig.add_hrect(y0=lo, y1=hi, fillcolor=meta["color"], opacity=0.09, line_width=0, layer="below")
        fig.add_annotation(
            x=0.005, xref="paper", y=(lo + hi) / 2, yref="y", text=name, showarrow=False,
            xanchor="left", font=dict(size=10, color=meta["color"]), opacity=0.9,
        )


def style_fig(fig: go.Figure, height: int = 420, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) if legend else None,
        showlegend=legend,
        hoverlabel=dict(font_size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def forecast_table(city: str, forecasts: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([{
        "City": city,
        "Forecast Date": f["forecast_date"],
        "Horizon (h)": f["horizon_hours"],
        "Predicted AQI": f["predicted_aqi"],
        "90% Interval Low": f["lower_bound"],
        "90% Interval High": f["upper_bound"],
        "CPCB Category": f["category"],
        "Data Source": "Live (Open-Meteo CAMS)" if is_live() else "Archive (CPCB/Kaggle)",
    } for f in forecasts])


def safety_tier(score: float) -> int:
    """0 = safe … 4 = hazardous, matching calculate_outdoor_safety_score verdict bands."""
    for tier, cutoff in enumerate([80.0, 60.0, 40.0, 20.0]):
        if score >= cutoff:
            return tier
    return 4


def cams_reference(city: str) -> Dict[str, float]:
    """CAMS physics-model daily AQI for upcoming dates (live mode only)."""
    if not is_live():
        return {}
    cams = CTX["bundle"]["cams_forecast"]
    rows = cams[cams["City"] == city].dropna(subset=["AQI"])
    return {str(d.date()): float(a) for d, a in zip(rows["Date"], rows["AQI"])}


def city_status(city: str) -> Dict[str, Any]:
    return U.get_latest_city_status(CTX["features"], city)


def city_forecasts(city: str) -> List[Dict[str, Any]]:
    return get_city_forecasts(city, CTX["mode"], CTX["stamp"])


def snapshot() -> pd.DataFrame:
    return get_snapshot(CTX["mode"], CTX["stamp"])


def now_label() -> str:
    return "Now" if is_live() else "Latest"


def city_pulse_strip(selected: str) -> None:
    snap = snapshot()
    cards = []
    for _, r in snap.sort_values("AQI").iterrows():
        sel = " sel" if r["City"] == selected else ""
        cards.append(
            f'<div class="as-city{sel}"><div class="n">{r["City"]}</div>'
            f'<div class="v">{r["AQI"]:.0f}</div>'
            f'<div class="c" style="color:{r["Color"] if r["Category"] not in ("Moderate", "Satisfactory") else "#475569"}">'
            f'{r["Category"]}</div><div class="b" style="background:{r["Color"]}"></div></div>'
        )
    st.html(f'<div class="as-pulse-grid">{"".join(cards)}</div>')


# -----------------------------------------------------------------------------
# PAGE: Dashboard
# -----------------------------------------------------------------------------
def hourly_outlook_card(city: str) -> None:
    outlook = L.hourly_outlook(CTX["bundle"]["hourly"], city, CTX["now"], hours_back=36, hours_ahead=48)
    if outlook.empty or outlook["PM2.5"].isna().all():
        return
    best = L.best_outdoor_window(outlook, CTX["now"])
    with card("hourly"):
        top = st.columns([3, 2], vertical_alignment="center")
        top[0].markdown("##### ⏱️ Hourly PM2.5 outlook")
        if best:
            top[1].html(
                f'<div class="as-tip" style="border-left-color:#16A34A;background:rgba(22,163,74,.08);">'
                f'🌿 <b>Best 2-hour window</b> (06–21 h): <b>{best["start"]:%a %H:%M}–{best["end"]:%H:%M}</b> · '
                f'PM2.5 ≈ {best["pm25"]:.0f} µg/m³</div>'
            )
        colors = [get_cpcb_category(v)["color"] if v is not None else "#CBD5E1" for v in outlook["PM2.5_index"]]
        fig = go.Figure(go.Bar(
            x=outlook["time"], y=outlook["PM2.5"], marker_color=colors,
            marker_opacity=[0.45 if f else 1.0 for f in outlook["is_forecast"]],
            customdata=np.where(outlook["is_forecast"], "CAMS forecast", "Model analysis"),
            hovertemplate="%{x|%a %d %b, %H:%M}<br>PM2.5 %{y:.0f} µg/m³<br>%{customdata}<extra></extra>",
        ))
        if best:
            fig.add_vrect(x0=best["start"] - pd.Timedelta(minutes=30), x1=best["end"] - pd.Timedelta(minutes=30),
                          fillcolor="#16A34A", opacity=0.12, line_width=0)
        fig.add_shape(type="line", x0=CTX["now"], x1=CTX["now"], y0=0, y1=1, yref="paper",
                      line=dict(color="rgba(15,23,42,.55)", width=1.5, dash="dash"))
        fig.add_annotation(x=CTX["now"], y=1, yref="paper", text="now", showarrow=False, yanchor="bottom",
                           font=dict(size=11))
        fig.add_hline(y=60, line_dash="dot", line_color="#EA580C",
                      annotation_text="India 24-h standard (60)", annotation_position="top left",
                      annotation_bgcolor="rgba(255,255,255,.85)", annotation_font_size=11)
        fig.update_yaxes(title="PM2.5 (µg/m³)", rangemode="tozero")
        fig.update_layout(bargap=0.15)
        st.plotly_chart(style_fig(fig, 300, legend=False), width="stretch")
        st.caption("Solid bars: recent hours · faded bars: CAMS hourly forecast. Bar colour = CPCB PM2.5 band.")


def page_dashboard() -> None:
    city, profile = st.session_state["city"], st.session_state["profile"]
    status = city_status(city)
    forecasts = city_forecasts(city)
    history = U.get_city_history(CTX["features"], city)
    cams = cams_reference(city)

    page_header("Overview", f"Air quality in {city}",
                ("Current conditions from live satellite-and-model data, with machine-learning forecasts for "
                 "the next 72 hours." if is_live() else
                 "Latest archived CPCB conditions and machine-learning forecasts for the following 72 hours."))
    city_pulse_strip(city)

    # Hero + forecast cards ----------------------------------------------------
    excluded = L.LIVE_AQI_EXCLUDED if is_live() else ()
    sub_indices = get_pollutant_sub_indices(status["latest_row"].to_dict())
    counted = [s for s in sub_indices if s["pollutant"] not in excluded]
    dominant = counted[0] if counted else None
    delta = status["aqi"] - status["prev_aqi"]
    arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "■")
    hero_label = ("AQI now · 24-h average" if is_live()
                  else f"Recorded AQI · {fmt_date(status['date'])}")

    hero_col, fc_col = st.columns([1.05, 1.95], gap="medium")
    with hero_col:
        color = status["color"]
        dominant_html = (
            f'<div class="as-muted" style="margin-top:10px;">Main pollutant: <b>{dominant["pollutant"]}</b> '
            f'({dominant["concentration"]} {dominant["unit"]})</div>' if dominant else ""
        )
        st.html(f"""
        <div class="as-hero" style="background:linear-gradient(135deg,{color}38 0%,{color}0A 70%),#FFFFFF;border-left:6px solid {color};">
          <div class="as-row"><span class="as-label">{hero_label}</span>{pill(status['category'], color)}</div>
          <div style="margin-top:14px;"><span class="as-num">{status['aqi']:.0f}</span><span class="as-unit">AQI</span></div>
          <div class="as-muted" style="margin-top:6px;">{arrow} {abs(delta):.0f} vs previous day ({status['prev_aqi']:.0f})</div>
          <div style="margin-top:12px;font-size:.92rem;">{status['description']}</div>
          {dominant_html}
          <div class="as-bar"><span style="width:{min(status['aqi'], 500) / 5:.1f}%;background:{color};"></span></div>
        </div>""")

    with fc_col:
        cols = st.columns(3, gap="small")
        for col, fc in zip(cols, forecasts):
            change = fc["predicted_aqi"] - status["aqi"]
            trend = "▲" if change > 0 else ("▼" if change < 0 else "■")
            ref = cams.get(fc["forecast_date"])
            ref_html = (f'<div class="as-ref">CAMS physics model: <b>{ref:.0f}</b> '
                        f'({get_cpcb_category(ref)["category"]})</div>' if ref is not None else "")
            with col:
                st.html(f"""
                <div class="as-fc" style="border-top:5px solid {fc['color']};">
                  <div class="as-row"><span class="as-label">{HORIZON_LABELS[fc['horizon_days']]}</span>
                    <span class="as-muted">{fmt_date(fc['forecast_date'], '%a %d %b')}</span></div>
                  <div class="as-big">{fc['predicted_aqi']:.0f}<span class="as-unit" style="font-size:.9rem;"> AQI</span></div>
                  {pill(fc['category'], fc['color'])}
                  <div class="as-muted" style="margin-top:10px;">90% range: <b>{fc['lower_bound']:.0f} – {fc['upper_bound']:.0f}</b></div>
                  <div class="as-muted">{trend} {abs(change):.0f} vs {now_label().lower()}</div>
                  {ref_html}
                </div>""")
        st.caption("Vayu ML forecast (LightGBM). Ranges are 90% prediction intervals calibrated on the model's "
                   "errors over the 2020 hold-out period"
                   + ("; the CAMS line is an independent physics-based forecast for comparison." if is_live() else "."))

    # KPI metrics --------------------------------------------------------------
    recent = history.tail(14)
    k = st.columns(5)
    k[0].metric("PM2.5 (µg/m³)", f"{status['pm25']:.0f}", f"{status['pm25'] - status['prev_pm25']:+.1f}",
                delta_color="inverse", border=True, chart_data=recent["PM2.5"].dropna().tolist(),
                help="Fine particulate matter (≤2.5 µm). WHO 24-h guideline: 15 µg/m³.")
    k[1].metric("PM10 (µg/m³)", f"{status['pm10']:.0f}", f"{status['pm10'] - status['prev_pm10']:+.1f}",
                delta_color="inverse", border=True, chart_data=recent["PM10"].dropna().tolist(),
                help="Coarse particulate matter (≤10 µm). WHO 24-h guideline: 45 µg/m³.")
    k[2].metric("Temperature (°C)", f"{status['temp']:.1f}", f"{status['temp'] - status['prev_temp']:+.1f}",
                delta_color="off", border=True, chart_data=recent["temp_mean"].dropna().tolist())
    k[3].metric("Humidity (%)", f"{status['humidity']:.0f}", border=True,
                chart_data=recent["humidity_mean"].dropna().tolist())
    k[4].metric("Max wind (km/h)", f"{status['wind_speed']:.0f}",
                f"{status['wind_speed'] - status['prev_wind_speed']:+.1f}", border=True,
                chart_data=recent["wind_speed_max"].dropna().tolist(),
                help="Stronger winds disperse pollutants, so an increase is usually good news.")

    # Trend chart --------------------------------------------------------------
    with card("trend"):
        top = st.columns([3, 2], vertical_alignment="center")
        top[0].markdown("##### 📈 Daily AQI trend & 72-hour forecast")
        options = [14, 30, 45] if is_live() else [30, 90, 180, 365]
        days = top[1].segmented_control(
            "Range", options, default=30 if is_live() else 90, required=True,
            format_func=lambda d: f"{d}d", key=f"dash_range_{CTX['mode']}", label_visibility="collapsed",
        )
        hist = history.tail(int(days))
        last_date = pd.to_datetime(status["date"])
        fc_dates = [last_date] + [pd.to_datetime(f["forecast_date"]) for f in forecasts]
        fc_vals = [status["aqi"]] + [f["predicted_aqi"] for f in forecasts]
        fc_hi = [status["aqi"]] + [f["upper_bound"] for f in forecasts]
        fc_lo = [status["aqi"]] + [f["lower_bound"] for f in forecasts]
        cams_pts = [(pd.to_datetime(f["forecast_date"]), cams[f["forecast_date"]])
                    for f in forecasts if f["forecast_date"] in cams]
        peak = max(float(hist["AQI"].max()), max(fc_hi), max([v for _, v in cams_pts], default=0.0))
        y_max = min(520.0, max(120.0, peak * 1.12))

        fig = go.Figure()
        add_cpcb_bands(fig, y_max)
        fig.add_trace(go.Scatter(
            x=hist["Date"], y=hist["AQI"], name="Observed AQI", mode="lines",
            line=dict(color=OBSERVED_COLOR, width=2.4), fill="tozeroy", fillcolor="rgba(37,99,235,0.06)",
            hovertemplate="%{x|%d %b %Y}<br>AQI %{y:.0f}<extra>Observed</extra>",
        ))
        fig.add_trace(go.Scatter(
            x=fc_dates + fc_dates[::-1], y=fc_hi + fc_lo[::-1], fill="toself", mode="lines",
            fillcolor="rgba(124,58,237,0.16)", line=dict(width=0), name="90% interval", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=fc_dates, y=fc_vals, name="ML forecast", mode="lines+markers",
            line=dict(color=FORECAST_COLOR, width=3, dash="dot"), marker=dict(size=9),
            hovertemplate="%{x|%d %b %Y}<br>Forecast %{y:.0f}<extra></extra>",
        ))
        if cams_pts:
            fig.add_trace(go.Scatter(
                x=[d for d, _ in cams_pts], y=[v for _, v in cams_pts], name="CAMS forecast", mode="markers",
                marker=dict(symbol="diamond-open", size=11, color="#0F172A", line=dict(width=2)),
                hovertemplate="%{x|%d %b %Y}<br>CAMS %{y:.0f}<extra></extra>",
            ))
        fig.add_shape(type="line", x0=last_date, x1=last_date, y0=0, y1=1, yref="paper",
                      line=dict(color="rgba(100,116,139,.6)", width=1, dash="dash"))
        fig.update_yaxes(range=[0, y_max], title="AQI")
        fig.update_xaxes(title=None)
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(style_fig(fig, 400), width="stretch")
        cpcb_legend()

    if is_live():
        hourly_outlook_card(city)

    # Advisory + pollutant breakdown -------------------------------------------
    left, right = st.columns([1.15, 1], gap="medium")
    with left:
        with card("advice"):
            tomorrow = forecasts[0]
            adv = get_health_advisory(tomorrow["predicted_aqi"], profile)["advice"]
            st.markdown(f"##### {PROFILES[profile]} Tomorrow's health advice")
            st.html(f'<div class="as-tip"><b>{tomorrow["category"]} ({tomorrow["predicted_aqi"]:.0f} AQI):</b> '
                    f'{adv["summary"]}</div>')
            a1, a2, a3 = st.columns(3)
            a1.markdown(f"**🏃 Activity**  \n{adv['activity']}")
            a2.markdown(f"**😷 Mask**  \n{adv['mask']}")
            a3.markdown(f"**🏠 Indoors**  \n{adv['indoor']}")

            cig = pm25_cigarette_equivalent(status["pm25"])
            who_ratio = status["pm25"] / WHO_24H_GUIDELINES["PM2.5"]
            st.divider()
            e1, e2 = st.columns(2)
            e1.metric("🚬 Cigarette equivalent", f"{cig:.1f} / day",
                      help="Berkeley Earth rule of thumb: breathing 22 µg/m³ PM2.5 for 24 h has a "
                           "health impact similar to smoking one cigarette.")
            e2.metric("🌍 PM2.5 vs WHO guideline", f"{who_ratio:.1f}×",
                      help="Ratio of the 24-hour PM2.5 level to the WHO 2021 guideline of 15 µg/m³.")

    with right:
        with card("drivers"):
            st.markdown("##### 🧪 What is driving the AQI?")
            if sub_indices:
                sdf = pd.DataFrame(sub_indices).sort_values("sub_index")
                sdf["label"] = [f"{p}*" if p in excluded else p for p in sdf["pollutant"]]
                sdf["bar_color"] = ["#CBD5E1" if p in excluded else c for p, c in zip(sdf["pollutant"], sdf["color"])]
                bar = go.Figure(go.Bar(
                    x=sdf["sub_index"], y=sdf["label"], orientation="h",
                    marker_color=sdf["bar_color"], text=sdf["sub_index"].round(0).astype(int),
                    textposition="outside", cliponaxis=False,
                    customdata=np.stack([sdf["concentration"], sdf["unit"], sdf["category"]], axis=1),
                    hovertemplate="<b>%{y}</b><br>Sub-index %{x:.0f} (%{customdata[2]})"
                                  "<br>Concentration %{customdata[0]} %{customdata[1]}<extra></extra>",
                ))
                bar.update_xaxes(range=[0, max(110.0, float(sdf["sub_index"].max()) * 1.25)], title="CPCB sub-index")
                st.plotly_chart(style_fig(bar, 290, legend=False), width="stretch")
                note = ("The AQI is set by the pollutant with the highest sub-index. "
                        + ("*Ozone is shown but not counted: the CAMS model is known to over-estimate surface "
                           "ozone over India." if is_live() else
                           "Estimated from daily means (CPCB uses 8-hour maxima for O₃ and CO)."))
                st.caption(note)
            else:
                st.info("No pollutant readings available for the latest day.")

    st.download_button(
        "Download forecast (CSV)", forecast_table(city, forecasts).to_csv(index=False).encode("utf-8"),
        file_name=f"vayu_forecast_{city.lower()}_{status['date']}.csv", mime="text/csv",
        icon=":material/download:",
    )


# -----------------------------------------------------------------------------
# PAGE: India map
# -----------------------------------------------------------------------------
def page_map() -> None:
    import folium
    from streamlit_folium import st_folium

    page_header("Overview", "India air quality map",
                "Colour-coded AQI across the eight monitored metropolitan hubs. Switch between current values "
                "and the 24–72 hour forecasts.", show_context=False)

    snap = snapshot()
    first = "Now" if is_live() else "Latest recorded"
    modes = {first: "AQI", "+24h forecast": "F24", "+48h forecast": "F48", "+72h forecast": "F72"}
    mode = st.segmented_control("Show", list(modes), default=first, required=True, key=f"map_mode_{CTX['mode']}")
    col = modes[mode]
    cat_col = "Category" if col == "AQI" else f"{col}_cat"
    color_col = "Color" if col == "AQI" else f"{col}_color"

    map_col, list_col = st.columns([1.45, 1], gap="medium")
    with map_col:
        with card("map"):
            fmap = folium.Map(location=[22.3, 79.0], zoom_start=5, tiles="OpenStreetMap",
                              control_scale=True, scrollWheelZoom=False)
            for _, r in snap.iterrows():
                loc = U.CITY_LOCATIONS.get(r["City"])
                if loc is None:
                    continue
                color, value = r[color_col], r[col]
                popup = f"""
                <div style="font-family:Inter,Arial,sans-serif;min-width:190px;">
                  <div style="font-size:15px;font-weight:700;margin-bottom:6px;">{r['City']}</div>
                  <div style="background:{color};color:{U.text_color_for(color)};padding:4px 8px;border-radius:6px;
                       font-weight:700;text-align:center;margin-bottom:8px;">{mode}: {value:.0f} · {r[cat_col]}</div>
                  <div style="font-size:12px;line-height:1.6;">
                    {first} ({fmt_date(r['Date'])}): <b>{r['AQI']:.0f}</b><br>
                    +24h / +48h / +72h: <b>{r['F24']:.0f} / {r['F48']:.0f} / {r['F72']:.0f}</b><br>
                    PM2.5 {r['PM2.5']:.0f} · PM10 {r['PM10']:.0f} µg/m³<br>
                    {r['Temp']:.1f} °C · wind {r['Wind']:.0f} km/h
                  </div>
                </div>"""
                icon_html = (
                    f'<div style="background:{color};color:{U.text_color_for(color)};width:46px;height:46px;'
                    f'border-radius:50%;display:flex;align-items:center;justify-content:center;font:700 14px Inter,Arial,sans-serif;'
                    f'border:3px solid #fff;box-shadow:0 2px 8px rgba(15,23,42,.35);">{value:.0f}</div>'
                )
                folium.Marker(
                    location=[loc["lat"], loc["lon"]],
                    icon=folium.DivIcon(html=icon_html, icon_size=(46, 46), icon_anchor=(23, 23)),
                    tooltip=f"{r['City']}: {value:.0f} ({r[cat_col]})",
                    popup=folium.Popup(popup, max_width=260),
                ).add_to(fmap)
            st_folium(fmap, height=540, use_container_width=True, returned_objects=[],
                      key=f"map_{CTX['mode']}_{col}")
        cpcb_legend()

    with list_col:
        with card("ranking"):
            st.markdown(f"##### 🏆 City ranking · {mode}")
            ranking = snap.sort_values(col)[["City", col, cat_col, "Trend"]].rename(
                columns={col: "AQI value", cat_col: "Category"})
            st.dataframe(
                ranking, hide_index=True, width="stretch", height=330,
                column_config={
                    "City": st.column_config.TextColumn(width=92),
                    "AQI value": st.column_config.NumberColumn("AQI", format="%.0f", width=48),
                    "Category": st.column_config.TextColumn(width=104),
                    "Trend": st.column_config.LineChartColumn("Trend", y_min=0, width=84),
                },
            )
            cleanest, worst = ranking.iloc[0], ranking.iloc[-1]
            st.markdown(f"🌿 Cleanest: **{cleanest['City']}** ({cleanest['AQI value']:.0f})  \n"
                        f"🏭 Most polluted: **{worst['City']}** ({worst['AQI value']:.0f})")
            n_bad = int((snap[col] > 200).sum())
            st.caption(f"{n_bad} of {len(snap)} cities are in the CPCB 'Poor' band or worse ({mode.lower()}).")

    with st.expander("Full city table"):
        table = snap[["City", "Date", "AQI", "Category", "F24", "F48", "F72", "PM2.5", "PM10", "Temp", "Wind"]]
        st.dataframe(
            table.rename(columns={"F24": "+24h", "F48": "+48h", "F72": "+72h", "Temp": "Temp (°C)",
                                  "Wind": "Wind (km/h)", "Date": "As of", "AQI": first}),
            hide_index=True, width="stretch",
        )


# -----------------------------------------------------------------------------
# PAGE: Should I go out?
# -----------------------------------------------------------------------------
CHECKLIST = {
    "relaxed": {
        "🚶 Morning walk": "Go ahead — conditions are fine.",
        "🏃 Intense cardio outdoors": "Safe outdoors.",
        "😷 Mask": "Not needed.",
        "🌬️ Air purifier": "Optional.",
        "🪟 Windows": "Keep open for ventilation.",
    },
    "caution": {
        "🚶 Morning walk": "Prefer late morning/afternoon; avoid busy roads.",
        "🏃 Intense cardio outdoors": "Keep it light outdoors, or move intense workouts indoors.",
        "😷 Mask": "Wear a well-fitted mask in traffic; N95 if you are sensitive.",
        "🌬️ Air purifier": "Recommended in bedrooms.",
        "🪟 Windows": "Ventilate around midday only; close during peak traffic.",
    },
    "strict": {
        "🚶 Morning walk": "Skip it — walk indoors instead.",
        "🏃 Intense cardio outdoors": "Indoors only.",
        "😷 Mask": "Certified N95/KN95 with a snug seal for any trip outside.",
        "🌬️ Air purifier": "Run HEPA filtration continuously.",
        "🪟 Windows": "Keep closed; wet-mop floors to trap dust.",
    },
}


def page_planner() -> None:
    city, profile = st.session_state["city"], st.session_state["profile"]
    status = city_status(city)
    forecasts = city_forecasts(city)

    page_header("Plan & protect", "Should I go out?",
                "A personal outdoor safety score for now and the next three days, adjusted for how "
                "sensitive your profile is to air pollution.")

    days = [{"label": now_label(), "date": status["date"], "aqi": status["aqi"], "observed": True}]
    days += [{"label": HORIZON_LABELS[f["horizon_days"]], "date": f["forecast_date"],
              "aqi": f["predicted_aqi"], "observed": False} for f in forecasts]
    for d in days:
        d["safety"] = calculate_outdoor_safety_score(d["aqi"], profile)
        d["cat"] = get_cpcb_category(d["aqi"])

    best = max(days[1:], key=lambda d: d["safety"]["safety_score"])
    gauge_col, cards_col = st.columns([1, 2.2], gap="medium")
    with gauge_col:
        with card("gauge"):
            today = days[0]["safety"]
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=today["safety_score"],
                number=dict(suffix="/100", font=dict(size=30), valueformat=".0f"),
                gauge=dict(
                    axis=dict(range=[0, 100], tickwidth=1),
                    bar=dict(color="#0F172A", thickness=0.22),
                    steps=[dict(range=[0, 20], color="#FF0000"), dict(range=[20, 40], color="#FF6600"),
                           dict(range=[40, 60], color="#FFC000"), dict(range=[60, 80], color="#92D050"),
                           dict(range=[80, 100], color="#00B050")],
                ),
                title=dict(text=f"Safety score · {now_label().lower()}", font=dict(size=14)),
            ))
            st.plotly_chart(style_fig(gauge, 250, legend=False), width="stretch")
            st.html(f'<div style="text-align:center;">{pill(today["verdict"], today["color"])}'
                    f'<div class="as-muted" style="margin-top:8px;">{today["recommendation"]}</div></div>')

    with cards_col:
        cols = st.columns(len(days), gap="small")
        for col, d in zip(cols, days):
            s = d["safety"]
            star = " ⭐" if d is best else ""
            with col:
                st.html(f"""
                <div class="as-fc" style="border-top:5px solid {s['color']};">
                  <div class="as-label">{d['label']}{star}</div>
                  <div class="as-muted">{fmt_date(d['date'], '%a %d %b')}</div>
                  <div class="as-big">{s['safety_score']:.0f}<span class="as-unit" style="font-size:.85rem;">/100</span></div>
                  {pill(s['verdict'], s['color'])}
                  <div class="as-muted" style="margin-top:10px;">AQI {d['aqi']:.0f} · {d['cat']['category']}</div>
                </div>""")
        st.success(f"**Best day in the next 72 hours:** {fmt_date(best['date'], '%A %d %b')} — safety score "
                   f"{best['safety']['safety_score']:.0f}/100 ({best['safety']['verdict']}).", icon="⭐")
        if is_live():
            outlook = L.hourly_outlook(CTX["bundle"]["hourly"], city, CTX["now"], hours_back=0, hours_ahead=30)
            window = L.best_outdoor_window(outlook, CTX["now"])
            if window:
                st.info(f"**Best time to step out (next 24 h, 06:00–21:00):** {window['start']:%A %H:%M}–"
                        f"{window['end']:%H:%M}, when PM2.5 is forecast at about {window['pm25']:.0f} µg/m³.",
                        icon="🕒")

    st.markdown("#### 📋 Your precautions checklist")
    choice = st.segmented_control(
        "Day", list(range(len(days))), default=0, required=True, key="planner_day",
        format_func=lambda i: f"{days[i]['label']} · {fmt_date(days[i]['date'], '%d %b')}",
    )
    d = days[int(choice)]
    # Use the stricter of the profile-adjusted safety tier and the CPCB category, so the
    # checklist never contradicts the category-based health advice shown beside it.
    tier = safety_tier(d["safety"]["safety_score"])
    tier_level = 0 if tier <= 1 else (1 if tier == 2 else 2)
    cat_level = {"Good": 0, "Satisfactory": 0, "Moderate": 1}.get(d["cat"]["category"], 2)
    group = ["relaxed", "caution", "strict"][max(tier_level, cat_level)]
    adv = get_health_advisory(d["aqi"], profile)["advice"]

    c1, c2 = st.columns([1.2, 1], gap="medium")
    with c1:
        with card("advice_detail"):
            st.markdown(f"**{PROFILES[profile]} Advice for {profile.lower()}** — AQI {d['aqi']:.0f} ({d['cat']['category']})")
            st.html(f'<div class="as-tip">{adv["summary"]}</div>')
            st.markdown(f"- **Activity:** {adv['activity']}\n- **Mask:** {adv['mask']}\n- **Indoors:** {adv['indoor']}")
    with c2:
        with card("checklist"):
            rows = "".join(f"<b>{k}</b><span>{v}</span>" for k, v in CHECKLIST[group].items())
            st.html(f'<div class="as-kv">{rows}</div>')

    with card("exposure"):
        st.markdown(f"##### 🫁 Current exposure in {city}")
        e = st.columns(4)
        e[0].metric("Cigarette equivalent", f"{pm25_cigarette_equivalent(status['pm25']):.1f}",
                    help="Berkeley Earth estimate: 22 µg/m³ of PM2.5 over 24 h ≈ 1 cigarette.")
        e[1].metric("PM2.5 vs WHO", f"{status['pm25'] / WHO_24H_GUIDELINES['PM2.5']:.1f}×",
                    help="WHO 2021 24-hour guideline: 15 µg/m³.")
        e[2].metric("PM10 vs WHO", f"{status['pm10'] / WHO_24H_GUIDELINES['PM10']:.1f}×",
                    help="WHO 2021 24-hour guideline: 45 µg/m³.")
        e[3].metric("Vulnerability factor", {"general": "1.00×", "outdoor_workers": "1.15×",
                                             "elderly": "1.25×", "asthma": "1.35×"}[normalize_user_group(profile)],
                    help="How strongly the safety score penalises pollution for your profile.")


# -----------------------------------------------------------------------------
# PAGE: Alert centre
# -----------------------------------------------------------------------------
def page_alerts() -> None:
    if not hasattr(st, "iframe"):
        import streamlit.components.v1 as components
    from src.alerts import (
        DEFAULT_ALERT_THRESHOLD,
        build_alert_message_html,
        build_alert_message_text,
        evaluate_and_dispatch_alerts,
        is_email_configured,
        is_telegram_configured,
    )

    page_header("Plan & protect", "Alert center",
                "Scan every city's 72-hour forecast against your warning threshold, preview the alert, "
                "and dispatch it by email or Telegram.", show_context=False)

    snap = snapshot()
    top = st.columns([2, 1, 1], gap="medium")
    with top[0]:
        default_threshold = int(min(450, max(50, round(DEFAULT_ALERT_THRESHOLD / 10) * 10)))
        threshold = st.slider("Warning threshold (AQI)", 50, 450, default_threshold, step=10, key="alert_threshold",
                              help="An alert fires when any forecast in the next 72 hours reaches this AQI. "
                                   "200 marks the start of CPCB 'Poor'.")
        st.caption(f"Threshold category: **{get_cpcb_category(threshold)['category']}**")
    email_ok, tg_ok = is_email_configured(), is_telegram_configured()
    top[1].markdown("**📧 Email channel**  \n" + (":green-badge[Connected]" if email_ok else ":gray-badge[Not configured]"))
    top[2].markdown("**✈️ Telegram channel**  \n" + (":green-badge[Connected]" if tg_ok else ":gray-badge[Not configured]"))

    scan = snap[["City", "AQI", "F24", "F48", "F72"]].copy()
    scan["Peak"] = scan[["F24", "F48", "F72"]].max(axis=1)
    scan["Status"] = np.where(scan["Peak"] >= threshold, "⚠️ Alert", "✅ Below")
    scan = scan.sort_values("Peak", ascending=False).rename(
        columns={"AQI": now_label(), "F24": "+24h", "F48": "+48h", "F72": "+72h", "Peak": "72h peak"})
    n_alerts = int((scan["Status"] == "⚠️ Alert").sum())

    with card("scan"):
        st.markdown(f"##### 🔎 Forecast scan · {n_alerts} of {len(scan)} cities reach {threshold} AQI within 72 h")
        aqi_cols = [now_label(), "+24h", "+48h", "+72h", "72h peak"]

        def _cell(v: float) -> str:
            c = get_cpcb_category(v)["color"]
            return f"background-color:{c};color:{U.text_color_for(c)};font-weight:600;"

        st.dataframe(scan.style.map(_cell, subset=aqi_cols).format("{:.0f}", subset=aqi_cols),
                     hide_index=True, width="stretch")

    st.markdown("#### ✉️ Compose alert")
    c1, c2 = st.columns([1, 1.6], gap="medium")
    with c1:
        alert_city = st.selectbox("City", CITIES, index=CITIES.index(st.session_state["city"]), key="alert_city")
        horizon = st.segmented_control("Forecast horizon", [1, 2, 3], default=1, required=True,
                                       format_func=lambda h: HORIZON_LABELS[h], key="alert_horizon")
        fc = city_forecasts(alert_city)[int(horizon) - 1]
        crosses = fc["predicted_aqi"] >= threshold
        st.html(f'<div class="as-card"><div class="as-label">Forecast for {fmt_date(fc["forecast_date"])}</div>'
                f'<div class="as-big">{fc["predicted_aqi"]:.0f} AQI</div>'
                f'{pill(fc["category"], fc["color"])}'
                f'<div class="as-muted" style="margin-top:10px;">{"Above" if crosses else "Below"} your threshold '
                f'of {threshold}.</div></div>')

        live_channels = email_ok or tg_ok
        if st.button("Send alert now" if live_channels else "Run simulated alert", type="primary",
                     icon=":material/send:", width="stretch"):
            with st.spinner("Dispatching…"):
                outcome = evaluate_and_dispatch_alerts(
                    city=alert_city, forecast_aqi=fc["predicted_aqi"], horizon_hours=fc["horizon_hours"],
                    threshold=float(threshold), force_send=True,
                )
            if live_channels:
                st.toast(f"Telegram: {'sent' if outcome['telegram_sent'] else 'not sent'} · "
                         f"Email: {'sent' if outcome['email_sent'] else 'not sent'}", icon="📨")
                if not (outcome["telegram_sent"] or outcome["email_sent"]):
                    st.error("Dispatch failed. Check your credentials in `.env` and the server log.")
                else:
                    st.success("Alert dispatched.")
            else:
                st.info("Simulation complete: the message on the right is exactly what would be sent. "
                        "Add SMTP or Telegram credentials to `.env` (see `.env.example`) for live delivery.")
        if not crosses:
            st.caption("This forecast is below your threshold; the scheduled pipeline would not send it "
                       "automatically, but you can still send it manually.")

    with c2:
        advisories = get_health_advisory(fc["predicted_aqi"])
        text_msg = build_alert_message_text(alert_city, fc["predicted_aqi"], fc["category"],
                                            fc["horizon_hours"], float(threshold), advisories)
        html_msg = build_alert_message_html(alert_city, fc["predicted_aqi"], fc["category"], fc["color"],
                                            fc["horizon_hours"], float(threshold), advisories)
        t1, t2 = st.tabs(["📧 Email preview", "💬 Telegram / SMS text"])
        with t1:
            # The email markup is generated by our own template (trusted content).
            if hasattr(st, "iframe"):
                st.iframe(html_msg, height=620)
            else:  # Streamlit versions before st.iframe
                components.html(html_msg, height=620, scrolling=True)
        with t2:
            st.code(text_msg, language=None, wrap_lines=True)


# -----------------------------------------------------------------------------
# PAGE: City comparison (historical archive)
# -----------------------------------------------------------------------------
def calendar_heatmap(city_df: pd.DataFrame, year: int) -> go.Figure:
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    series = city_df.set_index("Date")["AQI"].groupby(level=0).mean().reindex(days)
    start = days[0] - pd.Timedelta(days=days[0].weekday())
    week = np.asarray((days - start).days // 7)
    weekday = np.asarray(days.weekday)
    n_weeks = int(week.max()) + 1
    z = np.full((7, n_weeks), np.nan)
    text = np.full((7, n_weeks), "", dtype=object)
    for d, w, wd, v in zip(days, week, weekday, series.to_numpy()):
        z[wd, w] = v
        text[wd, w] = f"{d:%a %d %b %Y}<br>" + (f"AQI {v:.0f} · {get_cpcb_category(v)['category']}"
                                                  if not np.isnan(v) else "No data")
    month_starts = [pd.Timestamp(year=year, month=m, day=1) for m in range(1, 13)]
    fig = go.Figure(go.Heatmap(
        z=z, x=list(range(n_weeks)), y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        colorscale=cpcb_colorscale(), zmin=0, zmax=500, xgap=3, ygap=3,
        text=text, hovertemplate="%{text}<extra></extra>", colorbar=CPCB_COLORBAR,
    ))
    fig.update_xaxes(tickvals=[int((m - start).days // 7) for m in month_starts],
                     ticktext=[m.strftime("%b") for m in month_starts], showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return style_fig(fig, 280, legend=False)


def page_compare() -> None:
    page_header("Insights", "City comparison",
                "Long-term trends, seasonality, pollutant mix and the COVID-19 lockdown natural experiment, "
                "from six years of CPCB station records.", show_context=False, archive_only=True)

    f1, f2 = st.columns([2, 1.2], gap="medium")
    selected = f1.multiselect("Cities", CITIES, default=[c for c in ["Delhi", "Mumbai", "Bengaluru"] if c in CITIES],
                              key="compare_cities")
    years = sorted(raw_df["Date"].dt.year.unique().tolist())
    yr_from, yr_to = f2.select_slider("Years", options=years, value=(years[0], years[-1]), key="compare_years")
    if not selected:
        st.warning("Select at least one city to compare.")
        return

    data = raw_df[raw_df["City"].isin(selected) & raw_df["Date"].dt.year.between(yr_from, yr_to)].sort_values("Date")

    tabs = st.tabs(["📈 Trends", "🗓️ Seasonality", "🧪 Pollutants", "📅 Calendar", "🦠 Lockdown impact", "📋 Summary"])

    with tabs[0]:
        window = st.segmented_control("Smoothing", [7, 30, 90], default=30, required=True,
                                      format_func=lambda w: f"{w}-day average", key="compare_window")
        fig = go.Figure()
        peak = 0.0
        for c in selected:
            cd = data[data["City"] == c].set_index("Date")["AQI"].groupby(level=0).mean()
            rolled = cd.rolling(f"{int(window)}D", min_periods=max(3, int(window) // 4)).mean()
            if rolled.notna().any():
                peak = max(peak, float(rolled.max()))
            fig.add_trace(go.Scatter(x=rolled.index, y=rolled.values, name=c, mode="lines", line=dict(width=2),
                                     hovertemplate="%{x|%d %b %Y}<br>%{y:.0f}<extra>" + c + "</extra>"))
        y_max = min(520.0, max(150.0, peak * 1.1))
        add_cpcb_bands(fig, y_max)
        fig.update_yaxes(range=[0, y_max], title="AQI")
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(style_fig(fig, 430), width="stretch")
        st.caption("Shaded bands show CPCB categories. The sharp dip in spring 2020 is the nationwide lockdown.")

    with tabs[1]:
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly = data.groupby(["City", data["Date"].dt.month])["AQI"].mean().unstack()
        monthly = monthly.reindex(columns=range(1, 13))
        heat = go.Figure(go.Heatmap(
            z=monthly.values, x=months, y=monthly.index.tolist(), colorscale=cpcb_colorscale(),
            zmin=0, zmax=500, xgap=3, ygap=3, texttemplate="%{z:.0f}", colorbar=CPCB_COLORBAR,
            hovertemplate="%{y} · %{x}<br>Mean AQI %{z:.0f}<extra></extra>",
        ))
        heat.update_yaxes(autorange="reversed")
        st.plotly_chart(style_fig(heat, 120 + 46 * len(selected), legend=False), width="stretch")
        st.caption("Winter (Nov–Jan) temperature inversions trap pollutants near the ground; monsoon rain "
                   "(Jul–Sep) washes particulates out of the air.")

    with tabs[2]:
        p1, p2 = st.columns([1.3, 1], gap="medium")
        with p1:
            pollutant = st.segmented_control("Pollutant", ["PM2.5", "PM10", "NO2", "SO2", "O3", "CO"],
                                             default="PM2.5", required=True, key="compare_pollutant")
            unit = "mg/m³" if pollutant == "CO" else "µg/m³"
            box = px.box(data, x="City", y=pollutant, color="City", points=False)
            if pollutant in WHO_24H_GUIDELINES:
                box.add_hline(y=WHO_24H_GUIDELINES[pollutant], line_dash="dash", line_color="#16A34A",
                              annotation_text=f"WHO 24-h guideline ({WHO_24H_GUIDELINES[pollutant]:.0f})",
                              annotation_position="top left")
            box.update_yaxes(title=f"{pollutant} ({unit})")
            box.update_xaxes(title=None)
            st.plotly_chart(style_fig(box, 400, legend=False), width="stretch")
        with p2:
            st.markdown("**Which pollutant sets the AQI most often?**")
            share = get_dominant_pollutant_share()
            share = share[share["City"].isin(selected)]
            stack = px.bar(share, y="City", x="Share (%)", color="Pollutant", orientation="h",
                           category_orders={"Pollutant": ["PM2.5", "PM10", "NO2", "O3", "CO", "SO2", "NH3"]},
                           hover_data={"Days": True, "Share (%)": ":.1f"})
            stack.update_xaxes(range=[0, 100], title="Share of days (%)")
            stack.update_yaxes(title=None)
            st.plotly_chart(style_fig(stack, 360), width="stretch")
            st.caption("Dominant pollutant = highest CPCB sub-index on that day, all years.")

    with tabs[3]:
        cc1, cc2 = st.columns(2)
        cal_city = cc1.selectbox("City", selected, key="cal_city")
        cal_years = sorted(raw_df.loc[raw_df["City"] == cal_city, "Date"].dt.year.unique().tolist(), reverse=True)
        cal_year = cc2.selectbox("Year", cal_years, key=f"cal_year_{cal_city}")
        st.plotly_chart(calendar_heatmap(raw_df[raw_df["City"] == cal_city], int(cal_year)), width="stretch")
        cpcb_legend()

    with tabs[4]:
        impact = U.lockdown_impact(raw_df, selected)
        if impact.empty:
            st.info("Not enough 2019 and 2020 data for the selected cities.")
        else:
            st.markdown(f"Mean levels during the nationwide lockdown (**{fmt_date(U.LOCKDOWN_START)} – "
                        f"{fmt_date(U.LOCKDOWN_END)}**) versus the same dates in 2019.")
            bar = px.bar(impact, x="City", y="Change (%)", color="Metric", barmode="group", text="Change (%)",
                         color_discrete_map={"AQI": "#2563EB", "PM2.5": "#DB2777", "NO2": "#0891B2"})
            bar.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
            bar.add_hline(y=0, line_color="rgba(100,116,139,.7)")
            st.plotly_chart(style_fig(bar, 400), width="stretch")
            aqi_rows = impact[impact["Metric"] == "AQI"]
            if not aqi_rows.empty:
                best = aqi_rows.sort_values("Change (%)").iloc[0]
                st.success(f"Largest AQI improvement: **{best['City']}** ({best['Change (%)']:.0f}%). "
                           "NO₂ — mostly from traffic — fell the most, showing how much vehicle emissions contribute.")
            st.dataframe(impact, hide_index=True, width="stretch")

    with tabs[5]:
        summary = U.city_summary_stats(data, selected)
        st.dataframe(
            summary, hide_index=True, width="stretch",
            column_config={
                "Mean AQI": st.column_config.ProgressColumn("Mean AQI", min_value=0, max_value=500, format="%.0f"),
                "Good/Satisfactory Days (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "Poor or Worse Days (%)": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )
        st.download_button("Download filtered data (CSV)", data.to_csv(index=False).encode("utf-8"),
                           file_name=f"vayu_{'_'.join(c.lower() for c in selected)}_{yr_from}-{yr_to}.csv",
                           mime="text/csv", icon=":material/download:")


# -----------------------------------------------------------------------------
# PAGE: Why this forecast? (SHAP + what-if)
# -----------------------------------------------------------------------------
def _fmt_value(feature: str, value: float) -> str:
    if feature.startswith("City_"):
        return "yes" if value >= 0.5 else "no"
    return f"{value:.4g}"


@st.fragment
def whatif_section(city: str, mode: str) -> None:
    row = U.get_city_history(features_for(mode), city).iloc[-1:]
    latest = row.iloc[0]
    controls = [
        # (column, label, min, max, step)
        ("AQI", "Current AQI", 10.0, 500.0, 1.0),
        ("PM2.5", "PM2.5 (µg/m³)", 0.0, 500.0, 1.0),
        ("PM10", "PM10 (µg/m³)", 0.0, 600.0, 1.0),
        ("temp_mean", "Temperature (°C)", -5.0, 50.0, 0.5),
        ("humidity_mean", "Humidity (%)", 0.0, 100.0, 1.0),
        ("precipitation", "Rainfall (mm)", 0.0, 150.0, 0.5),
        ("wind_speed_max", "Max wind (km/h)", 0.0, 80.0, 0.5),
    ]

    def _default(col: str, lo: float, hi: float) -> float:
        v = latest.get(col, lo)
        return float(min(hi, max(lo, lo if pd.isna(v) else float(v))))

    def _reset() -> None:
        for col, _, lo, hi, _ in controls:
            st.session_state[f"wi_{mode}_{city}_{col}"] = _default(col, lo, hi)

    with card("whatif"):
        h1, h2 = st.columns([3, 1], vertical_alignment="center")
        h1.markdown("##### 🎛️ What-if simulator")
        h2.button("Reset to actual", on_click=_reset, icon=":material/restart_alt:", key=f"wi_reset_{mode}_{city}",
                  width="stretch")
        st.caption("Change current conditions and watch the forecasts respond. Lagged and rolling features are "
                   "held fixed, so this shows model sensitivity, not a guaranteed real-world outcome.")
        overrides: Dict[str, float] = {}
        cols = st.columns(4)
        for i, (col, label, lo, hi, step) in enumerate(controls):
            key = f"wi_{mode}_{city}_{col}"
            if key not in st.session_state:
                st.session_state[key] = _default(col, lo, hi)
            overrides[col] = cols[i % 4].slider(label, lo, hi, step=step, key=key)

        baseline = U.scenario_forecast(row, {}, multi_models, feature_cols)
        scenario = U.scenario_forecast(row, overrides, multi_models, feature_cols)
        out = st.columns(3)
        for step in U.HORIZONS:
            cat = get_cpcb_category(scenario[step])["category"]
            diff = scenario[step] - baseline[step]
            out[step - 1].metric(f"{HORIZON_LABELS[step]} · {cat}", f"{scenario[step]:.0f} AQI",
                                 f"{diff:+.1f} vs actual forecast" if abs(diff) >= 0.05 else None,
                                 delta_color="inverse", border=True)


def page_explain() -> None:
    city = st.session_state["city"]
    page_header("Insights", "Why this forecast?",
                "Live SHAP (Shapley value) attributions show exactly which inputs pushed this forecast up or "
                "down — computed on the fly for the selected city.")

    horizon = st.segmented_control("Horizon", U.HORIZONS, default=1, required=True,
                                   format_func=lambda h: HORIZON_LABELS[h], key="explain_horizon")
    horizon = int(horizon)
    exp = explain_city(city, horizon, CTX["mode"], CTX["stamp"])
    contrib: pd.DataFrame = exp["contributions"]
    fc = city_forecasts(city)[horizon - 1]

    ups = contrib[contrib["shap"] > 0].head(3)
    downs = contrib[contrib["shap"] < 0].head(3)

    def _phrase(df: pd.DataFrame) -> str:
        return ", ".join(f"**{r.label}** ({_fmt_value(r.feature, r.value)}, {r.shap:+.0f})" for r in df.itertuples())

    with card("explain_summary"):
        st.markdown(
            f"For **{city}** on **{fmt_date(fc['forecast_date'])}** the model predicts **{exp['prediction']:.0f} AQI** "
            f"({fc['category']}). It starts from the average model output of **{exp['base_value']:.0f}**"
            + (f"; the biggest upward pushes come from {_phrase(ups)}" if not ups.empty else "")
            + (f"; the biggest downward pulls come from {_phrase(downs)}." if not downs.empty else ".")
        )
        if abs(exp["prediction"] - fc["predicted_aqi"]) > 0.5:
            st.caption(f"The displayed forecast ({fc['predicted_aqi']:.0f}) is clipped to the valid AQI range.")
        if is_live():
            st.caption("Live mode: NO, NOx, NH₃ and the BTX gases are not in the live feed and use each city's "
                       "typical archived values.")

    top_n = 10
    top = contrib.head(top_n)
    rest = float(contrib["shap"].iloc[top_n:].sum())
    labels = ["Average forecast (base)"] + [f"{r.label} = {_fmt_value(r.feature, r.value)}" for r in top.itertuples()]
    labels += [f"{len(contrib) - top_n} other features", "Forecast"]
    values = [exp["base_value"]] + top["shap"].tolist() + [rest, 0.0]
    measures = ["absolute"] + ["relative"] * (len(top) + 1) + ["total"]
    text = [f"{exp['base_value']:.0f}"] + [f"{v:+.1f}" for v in top["shap"]] + [f"{rest:+.1f}", f"{exp['prediction']:.0f}"]

    wf = go.Figure(go.Waterfall(
        orientation="h", measure=measures, y=labels, x=values, text=text, textposition="outside",
        increasing=dict(marker=dict(color="#DC2626")), decreasing=dict(marker=dict(color="#16A34A")),
        totals=dict(marker=dict(color=FORECAST_COLOR)), connector=dict(line=dict(color="rgba(100,116,139,.4)")),
        hovertemplate="%{y}<br>%{text}<extra></extra>",
    ))
    running = np.cumsum([exp["base_value"]] + top["shap"].tolist() + [rest])
    x_hi = float(max(running.max(), exp["base_value"], exp["prediction"])) * 1.18
    x_lo = min(0.0, float(running.min()) * 1.18)
    wf.update_yaxes(autorange="reversed")
    wf.update_xaxes(title="AQI", range=[x_lo, x_hi])

    c1, c2 = st.columns([1.6, 1], gap="medium")
    with c1:
        with card("waterfall"):
            st.markdown(f"##### 🧮 Forecast breakdown · {HORIZON_LABELS[horizon]}")
            st.plotly_chart(style_fig(wf, 520, legend=False), width="stretch")
            st.caption("🔴 pushes AQI up (worse air) · 🟢 pulls AQI down (cleaner air) · 🟣 final forecast")
    with c2:
        with card("drivers_table"):
            st.markdown("##### 📋 Top drivers")
            table = contrib.head(15)[["label", "value", "shap"]].rename(
                columns={"label": "Feature", "value": "Value", "shap": "Impact (AQI)"})
            st.dataframe(
                table, hide_index=True, width="stretch", height=480,
                column_config={
                    "Value": st.column_config.NumberColumn(format="%.2f"),
                    "Impact (AQI)": st.column_config.NumberColumn(format="%+.1f"),
                },
            )

    whatif_section(city, CTX["mode"])

    with st.expander("🌐 Global explanations across the 2020 test set"):
        g1, g2 = st.tabs(["Beeswarm summary", "Mean |SHAP| importance"])
        for tab, name, note in [
            (g1, "shap_summary_beeswarm.png", "Each dot is one prediction. Red = high feature value, blue = low; "
                                               "position shows the push on the forecast."),
            (g2, "shap_feature_importance.png", "Average absolute contribution of each feature across test days."),
        ]:
            with tab:
                path = MODELS_DIR / name
                if path.is_file():
                    st.caption(note)
                    st.image(str(path), width="stretch")
                else:
                    st.info("Run `python -m src.explain` to generate this plot.")


# -----------------------------------------------------------------------------
# PAGE: Model performance (historical archive)
# -----------------------------------------------------------------------------
def page_performance() -> None:
    page_header("Insights", "Model performance",
                "How accurate are the forecasts? Every number here is measured on the January–June 2020 "
                "hold-out period, which the models never saw during training.", show_context=False,
                archive_only=True)

    test_preds = get_test_predictions()
    if test_preds.empty:
        st.warning("No hold-out data available.")
        return
    horizon_metrics = U.summarize_horizon_metrics(test_preds)

    cols = st.columns(3, gap="small")
    for col, (_, r) in zip(cols, horizon_metrics.iterrows()):
        with col:
            with card(f"perf_{int(r['horizon'])}"):
                st.markdown(f"**{HORIZON_LABELS[int(r['horizon'])]} forecast**")
                m1, m2, m3 = st.columns(3)
                m1.metric("MAE", f"{r['MAE']:.1f}", help="Mean absolute error in AQI points.")
                m2.metric("R²", f"{r['R2']:.2f}", help="Share of AQI variance explained.")
                m3.metric("Hit rate", f"{r['Category Accuracy (%)']:.0f}%",
                          help="Days where the predicted CPCB category matched the actual one. "
                               f"Within ±1 category: {r['Within ±1 Category (%)']:.1f}%.")

    with card("benchmark"):
        st.markdown("##### 🏆 Benchmark: 24-hour forecast models")
        if metrics_df.empty:
            st.info("Run `python -m src.train` to generate the benchmark table.")
        else:
            b1, b2 = st.columns([1.2, 1], gap="medium")
            with b1:
                st.dataframe(metrics_df, hide_index=True, width="stretch",
                             column_config={"R2": st.column_config.NumberColumn("R²", format="%.3f"),
                                            "MAE Improvement (%)": st.column_config.NumberColumn(format="%+.1f%%"),
                                            "RMSE Improvement (%)": st.column_config.NumberColumn(format="%+.1f%%")})
                naive = metrics_df[metrics_df["Model"].str.contains("Naive", case=False)]
                best_row = metrics_df.iloc[0]
                if not naive.empty:
                    st.caption(f"**{best_row['Model']}** cuts RMSE by **{best_row['RMSE Improvement (%)']:.1f}%** "
                               f"versus the naive 'tomorrow = today' baseline.")
            with b2:
                melted = metrics_df.melt(id_vars="Model", value_vars=["MAE", "RMSE"], var_name="Metric",
                                         value_name="AQI error")
                fig = px.bar(melted, x="AQI error", y="Model", color="Metric", barmode="group", orientation="h",
                             color_discrete_map={"MAE": "#2563EB", "RMSE": "#94A3B8"})
                fig.update_yaxes(title=None, autorange="reversed")
                st.plotly_chart(style_fig(fig, 280), width="stretch")

    with card("reality"):
        a1, a2, a3 = st.columns([1.2, 1.4, 1], vertical_alignment="center")
        a1.markdown("##### 🎯 Forecast vs reality")
        city = a2.selectbox("City", CITIES, index=CITIES.index(st.session_state["city"]), key="perf_city",
                            label_visibility="collapsed")
        horizon = int(a3.segmented_control("Horizon", U.HORIZONS, default=1, required=True,
                                           format_func=lambda h: f"+{h * 24}h", key="perf_horizon",
                                           label_visibility="collapsed"))
        sub = test_preds[(test_preds["City"] == city) & (test_preds["horizon"] == horizon)].sort_values("target_date")
        lo, hi = get_intervals().get(horizon, (0.0, 0.0))
        y_max = min(520.0, max(150.0, float(max(sub["actual"].max(), (sub["predicted"] + hi).max())) * 1.1))
        fig = go.Figure()
        add_cpcb_bands(fig, y_max)
        fig.add_trace(go.Scatter(
            x=pd.concat([sub["target_date"], sub["target_date"][::-1]]),
            y=pd.concat([(sub["predicted"] + hi).clip(upper=500), (sub["predicted"] + lo).clip(lower=0)[::-1]]),
            fill="toself", fillcolor="rgba(124,58,237,0.13)", line=dict(width=0), mode="lines", name="90% interval",
            hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=sub["target_date"], y=sub["actual"], name="Actual", mode="lines",
                                 line=dict(color=OBSERVED_COLOR, width=2.2)))
        fig.add_trace(go.Scatter(x=sub["target_date"], y=sub["predicted"], name="Forecast", mode="lines",
                                 line=dict(color=FORECAST_COLOR, width=2, dash="dot")))
        fig.update_yaxes(range=[0, y_max], title="AQI")
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(style_fig(fig, 400), width="stretch")
        coverage = float(((sub["error"] >= lo) & (sub["error"] <= hi)).mean() * 100) if len(sub) else float("nan")
        scores = U.summarize_city_metrics(test_preds, horizon)
        city_row = scores[scores["City"] == city]
        if not city_row.empty:
            r = city_row.iloc[0]
            st.caption(f"{city}, {HORIZON_LABELS[horizon]}: MAE **{r['MAE']:.1f}**, RMSE **{r['RMSE']:.1f}**, "
                       f"category hit rate **{r['Category Accuracy (%)']:.0f}%**, actual values inside the 90% "
                       f"interval **{coverage:.0f}%** of the time. The lockdown in late March 2020 is a hard "
                       "regime change that no model saw during training.")

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with card("city_error"):
            st.markdown(f"##### 🏙️ Error by city · +{horizon * 24}h")
            fig = px.bar(scores, x="MAE", y="City", orientation="h", text="MAE",
                         color="Mean AQI", color_continuous_scale=["#92D050", "#FFC000", "#FF6600"])
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
            fig.update_yaxes(autorange="reversed", title=None)
            fig.update_xaxes(title="Mean absolute error (AQI)", range=[0, float(scores["MAE"].max()) * 1.2])
            st.plotly_chart(style_fig(fig, 360, legend=False), width="stretch")
            st.caption("Cities with higher and more volatile pollution (e.g. Ahmedabad, Delhi) are harder to forecast.")
    with c2:
        with card("confusion"):
            st.markdown(f"##### 🧩 Category confusion · +{horizon * 24}h")
            hsub = test_preds[test_preds["horizon"] == horizon]
            cm = pd.crosstab(hsub["actual_category"], hsub["predicted_category"])
            labels = [lab for lab in U.CPCB_LABELS if lab in cm.index or lab in cm.columns]
            cm = cm.reindex(index=labels, columns=labels, fill_value=0)
            pct = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0) * 100
            heat = go.Figure(go.Heatmap(
                z=pct.values, x=labels, y=labels, colorscale="Blues", zmin=0, zmax=100,
                text=cm.values, texttemplate="%{text}", showscale=False,
                hovertemplate="Actual %{y}<br>Predicted %{x}<br>%{z:.0f}% of actual (%{text} days)<extra></extra>",
            ))
            heat.update_yaxes(autorange="reversed", title="Actual")
            heat.update_xaxes(title="Predicted")
            st.plotly_chart(style_fig(heat, 360, legend=False), width="stretch")
            st.caption("All cities. Cell numbers are day counts; shading is the share of each actual category.")


# -----------------------------------------------------------------------------
# PAGE: About & methodology
# -----------------------------------------------------------------------------
def page_about() -> None:
    page_header("About", "Project & methodology",
                "How Vayu turns six years of CPCB monitoring data into explainable forecasts, live "
                "predictions and personalised health guidance.", show_context=False)

    k = st.columns(4)
    k[0].metric("Cities", len(CITIES), border=True)
    k[1].metric("Archived daily records", f"{len(raw_df):,}", border=True)
    k[2].metric("Model features", len(feature_cols), border=True)
    k[3].metric("Forecast horizon", "72 h", border=True, help="Separate models for +24 h, +48 h and +72 h.")

    steps = [
        ("1", "Data ingestion", "Kaggle 'Air Quality Data in India' (CPCB stations, 2015–2020) plus Open-Meteo "
                                "historical weather: temperature, humidity, rain, wind."),
        ("2", "Cleaning", "Trimmed pre-commissioning gaps, capped values to the 0–500 AQI scale, time-weighted "
                          "interpolation of short gaps."),
        ("3", "Feature engineering", "Lags (1–7 days), rolling statistics (3–14 days), cyclical calendar "
                                     "encodings, PM ratios and one-hot city indicators."),
        ("4", "Modelling", "Chronological split (train ≤ Jun 2019, validate Jul–Dec 2019, test 2020). LightGBM "
                           "& XGBoost benchmarked against naive persistence and an LSTM."),
        ("5", "Live inference", "Hourly Open-Meteo (CAMS) pollutants → CPCB 24-h / 8-h averaging → daily AQI → the "
                                "same feature pipeline → live 24–72 h forecasts."),
        ("6", "Explain & act", "TreeSHAP attributions for every forecast, demographic advisories, safety scores "
                               "and email/Telegram alerts via GitHub Actions."),
    ]
    cols = st.columns(3, gap="small")
    for i, (num, title, body) in enumerate(steps):
        with cols[i % 3]:
            st.html(f'<div class="as-card" style="margin-bottom:12px;"><div class="as-eyebrow-w" '
                    f'style="color:#2563EB;opacity:1;">Step {num}</div>'
                    f'<div style="font-weight:700;font-size:1.05rem;margin:4px 0 6px 0;">{title}</div>'
                    f'<div class="as-muted" style="font-size:.9rem;">{body}</div></div>')

    m1, m2 = st.columns(2, gap="medium")
    with m1:
        with card("about_goal"):
            st.markdown("""
##### 🎯 Problem & goal
Air pollution is one of India's largest public-health risks. Vayu gives **24–72 hour forecasts**,
explains the drivers behind each forecast, and turns them into **actionable advice** for vulnerable groups.

##### 🌿 UN Sustainable Development Goals
- **SDG 3 – Good Health & Well-being:** early warnings help sensitive people avoid harmful exposure.
- **SDG 11 – Sustainable Cities:** city-level insight into seasonal and traffic-driven pollution.

##### 🔬 Atmospheric science highlights
- **Winter smog:** cold, still air creates temperature inversions that trap particulates.
- **Monsoon scavenging:** heavy rain washes aerosols out, so Jul–Sep is the cleanest season.
- **Lockdown experiment:** spring 2020 shows how much traffic and industry contribute.
""")
    with m2:
        with card("about_limits"):
            st.markdown("""
##### 📡 Live data
- **Source:** Open-Meteo air-quality API (Copernicus **CAMS** global model, ~45 km grid) and Open-Meteo weather,
  refreshed every 30 minutes. No API key needed.
- **AQI method:** CPCB rules — 24-hour means for PM2.5, PM10, NO₂, SO₂; 8-hour maxima for CO; at least three
  pollutants including a PM.

##### ⚠️ Honest limitations
- CAMS values are **model estimates**, not CPCB ground monitors, so they can differ from official station readings.
- CAMS over-estimates surface **ozone** over India, so O₃ is displayed but excluded from the live AQI.
- NO, NOx, NH₃ and BTX gases are not in the live feed; the model receives each city's typical archived values.
- The models were trained on 2015–2020 station data; live forecasts assume similar behaviour today.
- The what-if simulator holds lag features fixed; it shows model sensitivity, not causal effects.
""")


# -----------------------------------------------------------------------------
# Navigation, sidebar & data context
# -----------------------------------------------------------------------------
st.logo(str(ASSETS_DIR / "logo.svg"), size="large", icon_image=str(ASSETS_DIR / "icon.svg"))

if "city" not in st.session_state:
    st.session_state["city"] = "Delhi" if "Delhi" in CITIES else CITIES[0]
if "profile" not in st.session_state:
    st.session_state["profile"] = "General Public"
if "data_mode" not in st.session_state:
    st.session_state["data_mode"] = "live"

pages = {
    "Overview": [
        st.Page(page_dashboard, title="Dashboard", icon=":material/dashboard:", url_path="dashboard", default=True),
        st.Page(page_map, title="India AQI Map", icon=":material/map:", url_path="map"),
    ],
    "Plan & Protect": [
        st.Page(page_planner, title="Should I Go Out?", icon=":material/directions_walk:", url_path="planner"),
        st.Page(page_alerts, title="Alert Center", icon=":material/notifications_active:", url_path="alerts"),
    ],
    "Insights": [
        st.Page(page_explain, title="Why This Forecast?", icon=":material/psychology:", url_path="explain"),
        st.Page(page_compare, title="City Comparison", icon=":material/compare_arrows:", url_path="compare"),
        st.Page(page_performance, title="Model Performance", icon=":material/insights:", url_path="performance"),
    ],
    "About": [
        st.Page(page_about, title="Methodology", icon=":material/info:", url_path="about"),
    ],
}
current_page = st.navigation(pages, position="sidebar", expanded=True)


def resolve_context() -> Dict[str, Any]:
    """Chooses live or archive data for this run; falls back to the archive if live fails."""
    error: Optional[str] = None
    if st.session_state["data_mode"] == "live":
        bundle = get_live_bundle()
        if bundle.get("error") is None:
            return {"mode": "live", "features": bundle["features"], "stamp": str(bundle["fetched_at"]),
                    "now": bundle["now"], "bundle": bundle, "error": None}
        error = bundle["error"]
    return {"mode": "archive", "features": archive_features, "stamp": "archive",
            "now": pd.Timestamp(ARCHIVE_END), "bundle": None, "error": error}


with st.sidebar:
    st.segmented_control(
        "Data source", ["live", "archive"], key="data_mode", required=True, width="stretch",
        format_func=lambda m: "📡 Live" if m == "live" else "🗄️ 2015–20",
        help="Live: current Open-Meteo (CAMS) data, refreshed every 30 min. "
             "2015–20: the CPCB station archive the models were trained and tested on.",
    )
    st.selectbox("📍 City", CITIES, key="city")
    st.selectbox("🧑 Your profile", list(PROFILES), key="profile",
                 format_func=lambda p: f"{PROFILES[p]} {p}",
                 help="Advice and safety scores are tailored to how sensitive this group is to pollution.")

CTX.update(resolve_context())

with st.sidebar:
    side_status = city_status(st.session_state["city"])
    st.html(
        f'<div class="as-side-now" style="border-left:5px solid {side_status["color"]};">'
        f'<div class="as-label">{"Now" if is_live() else "Latest"} in {st.session_state["city"]}</div>'
        f'<div style="display:flex;align-items:baseline;gap:6px;margin:2px 0 6px 0;">'
        f'<span style="font-size:1.9rem;font-weight:800;">{side_status["aqi"]:.0f}</span>'
        f'<span class="as-muted">AQI</span></div>{pill(side_status["category"], side_status["color"])}</div>'
    )
    if is_live():
        st.caption(f"📡 Live data · updated {CTX['now']:%H:%M} IST · Open-Meteo / Copernicus CAMS · "
                   "Models: LightGBM multi-horizon")
    else:
        st.caption(f"🗄️ CPCB archive via Kaggle, up to {fmt_date(ARCHIVE_END)} · Models: LightGBM multi-horizon")

current_page.run()

st.html('<div class="as-footer">Vayu · Forecasts are for guidance only — follow official CPCB / local '
        'advisories in emergencies. · Built with Streamlit, LightGBM & SHAP</div>')

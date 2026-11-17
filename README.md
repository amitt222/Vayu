# 🌬️ Vayu: Air Quality Forecasting & Health Alert Platform

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Machine Learning](https://img.shields.io/badge/ML-LightGBM%20%7C%20XGBoost-blue?style=flat)](https://github.com)
[![Explainability](https://img.shields.io/badge/XAI-SHAP-success?style=flat)](https://shap.readthedocs.io)
[![UN SDG 3](https://img.shields.io/badge/UN%20SDG-3%20Good%20Health-4C9F38?style=flat)](https://sdgs.un.org/goals/goal3)
[![UN SDG 11](https://img.shields.io/badge/UN%20SDG-11%20Sustainable%20Cities-FD9D24?style=flat)](https://sdgs.un.org/goals/goal11)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **College Final Year / 3rd-Year CSE Project**  
> An end-to-end Machine Learning and Full-Stack Geospatial Web Platform providing 24 to 72-hour air quality forecasts, game-theoretic SHAP explainability, demographic-tailored health advisories, and automated multi-channel early warning alerts for 8 major Indian metropolitan hubs:  
> **Delhi, Mumbai, Kolkata, Chennai, Bengaluru, Hyderabad, Ahmedabad, and Jaipur.**

---

## 📌 Table of Contents
- [Live Demo & Video](#-live-demo--video)
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Machine Learning Benchmark](#-machine-learning-benchmark)
- [Model Explainability with SHAP](#-model-explainability-with-shap)
- [CPCB Regulatory Standards & Health Advisories](#-cpcb-regulatory-standards--health-advisories)
- [Project Structure](#-project-structure)
- [Quickstart: Local Installation](#-quickstart-local-installation)
- [Deployment & Automation](#-deployment--automation)
- [UN Sustainable Development Goals (SDGs)](#-un-sustainable-development-goals-sdgs)
- [Viva & Technical Defense Highlights](#-viva--technical-defense-highlights)

---

## 🚀 Live Demo & Video

- **🌐 Live Web App:** [https://vayu-india.streamlit.app](https://share.streamlit.io) *(Deployable on Streamlit Community Cloud)*
- **🎥 2-Minute Demo Video:** [Watch Project Walkthrough on YouTube](https://youtube.com) *(Insert your recording link here)*

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Data_Ingestion["1. Data Ingestion & API Layer"]
        A1["Kaggle India AQI Dataset<br/>(city_day.csv - 2015 to 2020)"]
        A2["Open-Meteo Weather API<br/>(Temp, Humidity, Rain, Wind)"]
    end

    subgraph Preprocessing_Pipeline["2. Preprocessing & Quality Assurance"]
        B1["Pre-commissioning Date Trimming"]
        B2["CPCB Regulatory Capping (0-500 Winsorization)"]
        B3["Time-Weighted Linear Interpolation"]
    end

    subgraph Feature_Engineering["3. Feature Engineering (59 Features)"]
        C1["Autoregressive Lags (t-1, t-2, t-3, t-7)"]
        C2["Shifted Rolling Statistics (3d, 7d, 14d)"]
        C3["Cyclical Calendar Encodings (sin/cos month, seasons)"]
    end

    subgraph ML_Benchmarking["4. Multi-Horizon ML Forecasting"]
        D0["Naive Persistence Baseline (t-1)"]
        D1["LightGBM Regressor (Best Model: R²=0.898)"]
        D2["XGBoost Regressor"]
        D3["PyTorch LSTM (Comparison)"]
    end

    subgraph Explainability["5. XAI with TreeSHAP"]
        E1["Global Feature Importance Ranking"]
        E2["Beeswarm Attribution Plots"]
        E3["Local Waterfall Explanations"]
    end

    subgraph Presentation_Alerts["6. Web Interface & Notification Engine"]
        F1["Streamlit Multi-Page UI"]
        F2["Folium Interactive Geospatial Map"]
        F3["'Should I Go Out?' Safety Score (0-100)"]
        F4["Automated Dispatch (SMTP HTML Email & Telegram Bot)"]
    end

    A1 & A2 --> B1 --> B2 --> B3
    B3 --> C1 & C2 & C3
    C1 & C2 & C3 --> D0 & D1 & D2 & D3
    D1 --> E1 & E2 & E3
    D1 --> F1 & F2 & F3 & F4
```

---

## ✨ Key Features

0. **📡 Live Data Mode (default):**
   - Pulls current hourly pollutants (Open-Meteo air-quality API, Copernicus **CAMS** model) and weather for all 8 cities — no API key needed, cached for 30 minutes.
   - Converts them to a CPCB AQI (24-h means for PM2.5/PM10/NO₂/SO₂, 8-h maximum for CO), rebuilds the exact training features and produces **live 24/48/72 h forecasts**, shown next to the CAMS physics-model forecast for comparison.
   - Hourly PM2.5 outlook with the **best 2-hour window** to go outside. A sidebar toggle switches to the 2015–2020 CPCB archive; the app falls back to it automatically if the live fetch fails.
   - CAMS over-estimates surface ozone over India, so O₃ is displayed but excluded from the live AQI.
1. **Multi-Horizon ML Forecasting (24h, 48h, 72h):**
   - Predicts multi-step future AQI with **calibrated 90% prediction intervals**, derived from the models' real errors on the 2020 hold-out period (not fixed margins).
2. **Dashboard with Pollutant Breakdown:**
   - Latest conditions with day-over-day deltas and sparklines, a 72h forecast chart over CPCB bands, the **dominant pollutant** via CPCB sub-indices, cigarette-equivalent and WHO-guideline exposure, and CSV export.
3. **Interactive National Geospatial Map:**
   - AQI-labelled markers for 8 cities, switchable between latest recorded and +24/48/72h forecasts, with a ranking table and 30-day sparklines.
4. **"Should I Go Out?" Planner (0–100):**
   - Safety scores for today and the next 3 days per vulnerability profile (*General Public, Asthma Patients, Elderly/Children, Outdoor Workers*), best-day recommendation and a precautions checklist.
5. **Live Explainability (SHAP) + What-If Simulator:**
   - Per-city TreeSHAP waterfall computed on demand with a plain-language summary, plus sliders to test how the forecast responds to pollution and weather changes.
6. **Model Performance Page:**
   - Hold-out MAE / R² / CPCB category hit rate per horizon, forecast-vs-actual charts with interval coverage, per-city error and a category confusion matrix.
7. **City Comparison & Insights:**
   - Smoothed trends, month × city heatmap, GitHub-style calendar heatmap, pollutant distributions vs WHO guidelines, dominant-pollutant share, and a **COVID-19 lockdown impact** analysis (2020 vs 2019).
8. **Alert Center (Email & Telegram):**
   - Scans every city's 72h forecast against a threshold, previews the HTML email and Telegram text, and dispatches (or simulates without credentials).
9. **Production MLOps & GitHub Actions:**
   - Daily scheduled cron workflow (`0 0 * * *`) that refreshes weather data, retrains models, evaluates performance, and auto-deploys updates.

---

## 🏆 Machine Learning Benchmark

All models were evaluated on the **holdout test set (Jan 2020 to Jun 2020, 1,456 samples)** using a strict chronological split to prevent temporal data leakage:

| Model Architecture | MAE (AQI pts) | RMSE | $R^2$ Score | MAPE (%) | MAE Improvement | RMSE Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **LightGBM Regressor** | **15.66** | **27.00** | **0.8977** | **13.25%** | **+14.98%** | **+20.56%** |
| 🥈 **XGBoost Regressor** | 15.70 | 26.90 | 0.8984 | 13.28% | +14.77% | +20.86% |
| 🥉 **Naive Persistence ($t-1$)** | 18.42 | 33.99 | 0.8378 | 13.97% | Baseline (0.0%) | Baseline (0.0%) |
| 🔬 **PyTorch LSTM (Sequence)** | 19.87 | 31.68 | 0.8591 | 17.13% | -7.87% | +6.80% |

---

## 🔍 Model Explainability with SHAP

TreeSHAP reveals the atmospheric factors influencing forecasts:

- **Primary Inertia:** `AQI` and `PM2.5` from previous days carry the highest predictive weight.
- **Atmospheric Dispersion:** `wind_speed_max` and `precipitation` exhibit strong **negative SHAP values** (high wind and rain wash out particulates, reducing predicted AQI).
- **Combustion Proxies:** `CO` and `NOx` act as leading vehicular and industrial combustion indicators.

---

## 🩺 CPCB Regulatory Standards & Health Advisories

Vayu adheres to the **Central Pollution Control Board (CPCB) NAQI** standards:

| Category | AQI Range | Color Code | General Public Advice | Asthma / Vulnerable Groups |
| :--- | :---: | :---: | :--- | :--- |
| **Good** | 0 – 50 | `#00B050` | Clean air; ideal for all outdoor activities. | No restrictions. |
| **Satisfactory** | 51 – 100 | `#92D050` | Minor irritation in rare sensitive groups. | Normal outdoor activity; stay hydrated. |
| **Moderate** | 101 – 200 | `#FFC000` | Breathing discomfort in heavy traffic. | Reduce intense cardio; wear mask in traffic. |
| **Poor** | 201 – 300 | `#FF6600` | Unhealthy air; throat & eye irritation. | Avoid outdoor workouts; N95 mask required. |
| **Very Poor** | 301 – 400 | `#FF0000` | Respiratory distress on prolonged exposure. | Stay strictly indoors; keep inhalers ready. |
| **Severe** | 401 – 500+ | `#990033` | Emergency hazard; affects healthy people. | Sealed indoor rooms; N99/P100 respirator. |

---

## 📁 Project Structure

```
vayu/
├── .github/workflows/
│   └── retrain_and_refresh.yml   # Daily GitHub Actions automated retraining
├── .streamlit/
│   └── config.toml               # Streamlit Cloud custom emerald UI theme
├── app/
│   ├── app.py                    # Multi-page Streamlit web app
│   └── utils.py                  # Inference, forecasting, and geospatial helpers
├── data/
│   ├── raw/                      # Kaggle city_day.csv (excluded from git)
│   └── processed/                # Cleaned datasets and Open-Meteo weather cache
├── models/                       # Serialized LightGBM, XGBoost, and SHAP artifacts
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb # Comprehensive Jupyter EDA notebook
│   ├── generate_eda_plots.py     # Static high-res chart generator
│   └── plots/                    # Exported EDA visualizations
├── reports/figures/              # Exported SHAP beeswarm and bar charts
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Data loading, validation, and city filters
│   ├── preprocessing.py          # Outlier capping & time-series interpolation
│   ├── features.py               # Lags, rolling windows, and Open-Meteo API
│   ├── train.py                  # Chronological training & benchmark pipeline
│   ├── evaluate.py               # Performance metrics (MAE, RMSE, R²)
│   ├── explain.py                # TreeSHAP explainability routines
│   ├── advisory.py               # CPCB categories & health risk scoring
│   └── alerts.py                 # Multi-channel SMTP and Telegram notifications
├── .env.example                  # Environment secrets template
├── .gitignore                    # Git tracking rules
├── requirements.txt              # Production dependency specifications
└── README.md                     # Comprehensive project documentation
```

---

## 💻 Quickstart: Local Installation

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/amitt222/Vayu.git
cd vayu
python -m venv venv

# Activate Virtualenv (Windows)
venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
```bash
cp .env.example .env
# Edit .env with your optional Telegram / Gmail App Password credentials
```

### 3. Run Pipeline Steps
```bash
# Test data loading
python -m src.data_loader

# Run cleaning & interpolation
python -m src.preprocessing

# Generate features & fetch Open-Meteo weather
python -m src.features

# Train and benchmark models
python -m src.train

# Compute SHAP explanations
python -m src.explain

# Launch the Streamlit Web Application
streamlit run app/app.py
```

---

## 🌐 Deployment & Automation

### Streamlit Community Cloud
1. Push this repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io) and create a **New app**.
3. Select repository and set Main file path to: `app/app.py`.
4. Configure optional notification tokens under **Settings $\rightarrow$ Secrets**.

### GitHub Actions Automation
- The `.github/workflows/retrain_and_refresh.yml` workflow automatically runs every day at 00:00 UTC (05:30 IST) to fetch latest weather, retrain models, update forecasts, and push refreshed artifacts back to GitHub.

---

## 🌍 UN Sustainable Development Goals (SDGs)

- **Target 3.9 (Good Health & Well-Being):** Substantially reduce the number of deaths and illnesses from hazardous air pollution through predictive, demographic-tailored advisories.
- **Target 11.6 (Sustainable Cities & Communities):** Reduce the adverse per capita environmental impact of cities by improving municipal air quality tracking and public awareness.

---

## 🎓 Viva & Technical Defense Highlights

1. **Why use a chronological split instead of K-Fold cross validation?**  
   *Air pollution is an autoregressive time-series process. Shuffling records leaks future values into the past, artificially inflating $R^2$ while failing completely when deployed on real future data.*
2. **Why cap AQI at 500 instead of standard 3-sigma outlier deletion?**  
   *Deleting rows introduces gaps in daily time sequences, breaking lag features ($t-1, t-7$). Capping at 500 aligns with regulatory CPCB definitions, preserves sequence continuity, and prevents gradient explosion in tree splits.*
3. **Why did LightGBM outperform LSTM in this setup?**  
   *For tabular daily aggregate time-series enriched with explicit lags, moving averages, cyclical month encodings, and weather features, gradient boosted decision trees construct sharp decision boundaries more effectively with lower data requirements than recurrent neural networks.*

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).

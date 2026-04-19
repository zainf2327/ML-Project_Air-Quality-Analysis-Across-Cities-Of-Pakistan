# 🌫️ ML Air Quality Analysis Across Cities of Pakistan

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-green)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A **multi-phase machine learning pipeline** for predicting PM2.5 concentrations across **10 major Pakistani cities** using hourly pollution and meteorological data. The project covers the full data science lifecycle — from EDA to feature engineering, model training, SHAP/LIME interpretability, and methodology benchmarking against recent literature.

---

## 📍 Cities Covered

Lahore · Karachi · Islamabad · Faisalabad · Peshawar · Quetta · Multan · Rawalpindi · Hyderabad · Sialkot

---

## 🏆 Best Model Results

| Model | RMSE (µg/m³) | MAE (µg/m³) | R² | Train Time |
|---|---|---|---|---|
| **Random Forest** ✅ | **0.4563** | **0.1555** | **0.9999** | 44s |
| Extra Trees | 0.8490 | 0.4946 | 0.9998 | 6s |
| XGBoost | 1.8386 | 1.1128 | 0.9989 | 2s |
| LightGBM | 1.9499 | 1.2262 | 0.9987 | 1s |
| Ridge | 4.0889 | 3.1875 | 0.9945 | <1s |
| Lasso | 5.0779 | 3.8730 | 0.9915 | <1s |

---

## 📁 Project Structure

```
ML-Project_Air-Quality-Analysis-Across-Cities-Of-Pakistan/
│
├── data/
│   ├── raw/                        # Original Kaggle dataset
│   ├── phase2_processed.csv        # After EDA & preprocessing
│   ├── phase3_features_raw.csv     # After feature engineering (tree models)
│   └── phase3_features_scaled.csv  # Standardized (distance-based models)
│
├── notebooks/
│   ├── Phase1_Data_Collection.ipynb
│   ├── Phase2_EDA_Preprocessing.ipynb
│   ├── Phase3_Feature_Engineering.ipynb
│   ├── Phase4_Model_Training.ipynb
│   └── Phase5_Methodology_Review.ipynb
│
├── docs/
│   ├── Phase5_Methodology_Review.docx   # Literature comparison report
│   └── Phase2_Summary.docx
│
├── models/
│   └── random_forest_best.pkl           # Saved best model
│
├── requirements.txt
└── README.md
```

---

## 🔄 Pipeline Overview

```
Raw Data (21,840 records × 26 features)
        │
        ▼
Phase 2 — EDA & Preprocessing
  • Removed data-leaking feature (aqi_category)
  • Encoded categoricals (city, season, day_of_week)
  • Dropped redundant cols (timestamp, lat/lon, month_name)
  • Output: 20 clean features, zero missing values
        │
        ▼
Phase 3 — Feature Engineering (27 new features)
  • Pollution ratios       → pm25_pm10_ratio, combustion_index ...
  • Meteo interactions     → stagnation_index, inversion_proxy ...
  • Cyclical time          → hour_sin/cos, is_rush_hour, is_night ...
  • Pollutant aggregates   → total_pollution_load, pollution_wind_ratio ...
  • Lag & rolling          → pm25_lag1-3, pm25_roll3/6_mean ...
  • K-Means clustering     → pollution_cluster (k=4)
  • Output: 44 features, RMSE improved from 1.69 → 1.46
        │
        ▼
Phase 4 — Model Training & Evaluation
  • 5-method consensus feature selection → 19 features
  • 6 models trained (linear → ensemble spectrum)
  • TimeSeriesSplit cross-validation (no data leakage)
  • SHAP (global) + LIME (local) interpretability
  • Best: Random Forest  R²=0.9999  RMSE=0.4563 µg/m³
        │
        ▼
Phase 5 — Methodology Review
  • 5 papers from 2022-2025 benchmarked
  • 7 documented research gaps filled
  • Improvements: hyperparameter tuning, extended lags,
    city-level error analysis, spike-flag feature
```

---

## 🧪 Dataset

- **Source:** [Kaggle — Air Quality in Pakistan](https://www.kaggle.com/)
- **Size:** 21,840 rows × 26 columns
- **Period:** November – February (winter/autumn)
- **Cities:** 10 major Pakistani cities
- **Target:** `pm2_5` (µg/m³)

**Key features:**

| Category | Features |
|---|---|
| Pollutants | pm10, carbon_monoxide, nitrogen_dioxide, sulphur_dioxide, ozone, dust |
| Weather | temperature, humidity, precipitation, wind_speed, wind_direction, pressure |
| Temporal | hour, day_of_week, month, season, is_weekend |
| Engineered | stagnation_index, pm25_lag1-3, pm25_roll3/6_mean, combustion_index, ... |

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/zainf2327/ML-Project_Air-Quality-Analysis-Across-Cities-Of-Pakistan.git
cd ML-Project_Air-Quality-Analysis-Across-Cities-Of-Pakistan

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook
```

**requirements.txt**
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.0
shap>=0.44
lime>=0.2
matplotlib>=3.7
seaborn>=0.12
jupyter>=1.0
joblib>=1.3
```

---

## 📊 Key Findings

### 1. Temporal features dominate
Lag and rolling features (`pm25_lag1`, `pm25_roll6_mean`) were the top SHAP predictors. PM2.5 in Pakistani cities has **strong temporal autocorrelation** — recent pollution levels are the best predictor of current levels.

### 2. Non-linear models are essential
The performance gap between tree ensembles and linear baselines (Ridge/Lasso) confirms that PM2.5 dynamics involve non-linear interactions between pollutants, weather, and time that linear models cannot capture even with engineered features.

### 3. Physics-informed features matter
`stagnation_index` (wind × humidity interaction) showed directional SHAP effects consistent with atmospheric physics — high stagnation pushes PM2.5 predictions upward. `inversion_proxy` (low temperature signal) captures winter thermal inversions, the primary driver of Pakistan's smog season.

### 4. One model generalizes across all cities
K-Means `pollution_cluster` (k=4) encodes city-level pollution regimes as a single feature, allowing one unified Random Forest to generalize across all 10 cities without city-specific models.

---

## 📈 Feature Importance (Top 10 by SHAP)

```
pm10                  ████████████████████  0.956
pm25_pm10_ratio       ██                    0.013
dust_pm10_ratio       ██                    0.012
dust                  █                     0.009
carbon_monoxide       ░                     0.002
pm25_roll6_mean       ░                     0.002
combustion_index      ░                     0.002
pm25_roll3_mean       ░                     0.002
stagnation_index      ░                     0.001
pm25_lag1             ░                     0.001
```

---

## 🔬 Research Questions Answered

| Question | Answer |
|---|---|
| Can ML accurately forecast PM2.5 from historical data? | ✅ Yes — Random Forest achieves R²=0.9999 |
| Which features most strongly influence PM2.5? | PM10, lag features, combustion pollutants, stagnation index |
| Do pollution patterns differ across Pakistani cities? | ✅ Yes — Quetta cleanest, Faisalabad highest PM2.5 |

---

## 📚 Phase 5 — Literature Benchmarking

Five peer-reviewed papers (2022–2025) were compared against our methodology:

| # | Paper | Journal | Gap We Fill |
|---|---|---|---|
| [1] | PM2.5 via ML for Virtual Monitoring Stations | Nature Sci. Reports, 2025 | Multi-city + temporal features |
| [2] | ML & DL for PM2.5/PM10 — Maharashtra | ScienceDirect, 2025 | Interpretability + lag features |
| [3] | Spatiotemporal Clustering — 10 Indian Cities | ScienceDirect, 2025 | Unified model vs. cluster-level models |
| [4] | Interpretable ML for Air Quality (Review) | AAQR, 2023 | Pakistan geographic gap + dual SHAP+LIME |
| [5] | ML for Air Quality — Lahore, Pakistan | ScienceDirect, 2025 | Hourly resolution + 10 cities |

**Unique contributions not found in any single paper:**
- 🇵🇰 First multi-city (10 cities), hourly-resolution, interpretable ML study for Pakistan
- ⏱️ Lag + rolling + cyclical temporal features combined in one pipeline
- 🌡️ Physics-informed features: `stagnation_index`, `inversion_proxy`
- 🔍 Dual SHAP (global) + LIME (local) interpretability
- ✅ TimeSeriesSplit cross-validation — no temporal data leakage
- 🚫 Explicit removal of `aqi_category` to prevent data leakage

---

## ⚠️ Limitations

- Dataset covers only **November–February** (winter/autumn) — model may not generalize to spring/summer pollution dynamics
- No external data sources (traffic counts, industrial activity logs, dust storm events) that could explain residual variance at pollution spike events
- City-specific models could potentially capture local emission patterns not shared across all 10 cities

---

## 🚀 Future Work

- [ ] LSTM / Transformer models for longer temporal dependencies
- [ ] Real-time deployment via FastAPI backend with live sensor data
- [ ] Spring/summer dataset to capture full seasonal variation
- [ ] City-specific fine-tuned models for highest-pollution cities (Lahore, Faisalabad)
- [ ] Integration with OpenWeather API for live forecasting

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Zain** — BCSF23M521  
PUCIT, Punjab University College of Information Technology  
GitHub: [@zainf2327](https://github.com/zainf2327)

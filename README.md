# CareCast: Predictive Forecasting of Care Load & Placement Demand

> **"Forecasting capacity before crisis strikes"**

An end-to-end predictive analytics, machine learning forecasting, and capacity risk simulation platform designed for the **Unaccompanied Alien Children (UAC) Program** administered by the **U.S. Department of Health and Human Services (HHS)**, **Office of Refugee Resettlement (ORR)**, in coordination with **U.S. Customs and Border Protection (CBP)**.

---

## 🌟 Executive Overview

Under the *William Wilberforce Trafficking Victims Protection Reauthorization Act of 2008 (TVPRA)*, unaccompanied children encountered at the border must be transferred from CBP custody to HHS care within 72 hours. Managing shelter capacity, staffing, and child welfare requires proactive forecasting rather than reactive crisis response.

This project delivers:
1. **Multi-Horizon Care Load Projections**: Forward forecasts of children in active HHS shelter care (3, 7, 14, 30, and 60 days) with empirical 80% and 95% uncertainty intervals.
2. **Sponsor Discharge & Placement Velocity Modeling**: Demand forecasting for sponsor vetting, background checks, and safe releases.
3. **Hydrodynamic Flow & Net Pressure Dynamics**: Integration of upstream intake velocities, CBP buffer holding, transfers to HHS, and discharges.
4. **Early-Warning & Capacity Breach Engine**: Probabilistic Monte Carlo risk simulation delivering **10 to 25 days of advance surge lead time** before shelter bed capacity thresholds are breached.
5. **Interactive Executive Web Cockpit**: A Streamlit dashboard engineered with a clean, high-contrast light theme (`#FFFFFF` background, pure `#000000` typography) and Plotly visualizations.

---

## 📂 Repository Architecture

```text
care_load_forecasting/
├── .streamlit/
│   └── config.toml               # Light theme configuration (#FFFFFF background, #000000 text)
├── assets/
│   └── carecast_logo.svg         # Official CareCast shelter + forecast SVG logo
├── data/
│   ├── uac_care_load_data.csv    # Official daily operational dataset (Jan 2023 - Dec 2025)
│   ├── cleaned_data.csv          # Re-indexed, interpolated, and sanitized daily calendar
│   └── features.csv              # 56 engineered lag, rolling, and cyclical flow features
├── reports/
│   ├── care_load_forecasting_research_paper.md   # Comprehensive academic research paper
│   ├── research_paper_outline.md                 # Research outline & methodology
│   ├── executive_summary_hhs_stakeholders.md     # 1-page executive decision memo
│   ├── executive_summary.md                      # Executive brief
│   ├── model_comparison.csv                      # Multi-metric benchmark results across models
│   └── kpi_summary.csv                           # Final operational KPIs
├── src/
│   ├── __init__.py               # Package init
│   ├── data_prep.py              # Schema cleaning, interpolation & seasonal decomposition
│   ├── feature_engineering.py    # Lag creation, net pressure momentum & calendar encodings
│   ├── validation.py             # Temporal train/test splitting & walk-forward validation
│   ├── models.py                 # Naive, MA, SARIMAX, Holt-Winters, Random Forest, GBDT
│   └── evaluate.py               # MAE, RMSE, MAPE, Accuracy, Surge Lead Time, Breach Risk
├── tests/
│   └── test_pipeline.py          # Comprehensive unit and integration test suite
├── app.py                        # Streamlit web dashboard (4 interactive tabs)
├── branding.py                   # Reusable CareCast branding header & SVG loader (Light & Dark Brown support)
├── run_pipeline.py               # End-to-end command-line execution runner
├── requirements.txt              # Production dependencies
└── README.md                     # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.10, 3.11, 3.12, 3.14)
- Git & Virtual Environment tool (`venv` or `conda`)

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-org/care_load_forecasting.git
cd care_load_forecasting

# Create and activate a virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quickstart & Usage

### 1. Run the End-to-End Pipeline
Execute the full data preparation, feature engineering, model training, and tournament evaluation pipeline:
```bash
python run_pipeline.py
```
This script will:
- Ingest and sanitize `data/uac_care_load_data.csv`.
- Reindex missing reporting days using continuous time interpolation.
- Generate 56 hydrodynamic features and save `data/features.csv`.
- Train and evaluate 6 models using expanding-window walk-forward validation.
- Output benchmark reports to `reports/model_comparison.csv` and `reports/kpi_summary.csv`.

### 2. Launch the Interactive Web Dashboard
```bash
python -m streamlit run app.py
```
*(Or `streamlit run app.py` if your environment has the script in PATH).*
Open your browser and navigate to `http://localhost:8501`.

### 3. Run Automated Unit Tests
```bash
python -m unittest tests/test_pipeline.py
```

---

## 📊 Dashboard Capabilities & Dual Theme Engine

CareCast features a real-time **Theme Switcher** in the sidebar supporting two distinct design systems:
- **☀️ Light Mode**: Pure white background (`#FFFFFF`), high-contrast black typography (`#000000`), slate borders, and crisp light cards.
- **☕ Dark Mode (Warm Espresso)**: Deep warm roasted espresso background (`#1C1613`), warm mocha surfaces (`#28201B`), warm brown borders (`#44352D`), cream typography (`#F5ECE5`), and warm amber/bronze accent highlights (`#E09F5A` — **strictly zero blue elements in dark mode**).

### Interactive Tabs:

1. **Care Load Forecast**:
   - Multi-step forward projection (up to 30+ days) with 95% confidence intervals.
   - Interactive capacity threshold selector with breach warnings and surge lead time.
   - Historical actuals vs. model fit overlay with theme-adapted palettes.
2. **Discharge Demand & Inflow Analysis**:
   - Dynamic Inflow vs. Outflow flow balance chart.
   - Net Pressure momentum indicator ($\text{Transfers} - \text{Discharges}$).
   - Placement demand volume projections.
3. **Model Comparison & Tournaments**:
   - Interactive leaderboard sorted by MAE, RMSE, MAPE, and Accuracy (%).
   - Multi-model horizon comparison bar charts.
   - Interactive overlay on test ground truth.
4. **Scenario & Stress Testing**:
   - Side-by-side operational scenario simulation with independent capacity thresholds and models.
   - Scenario variance and percentage divergence table.

---

## 🔬 Modeling Tournament Summary

Models evaluated across 3-day, 7-day, and 14-day walk-forward validation horizons:

| Model Paradigm | Algorithm | Target: Care Load Accuracy | Target: Discharge Accuracy | Surge Lead Time |
| :--- | :--- | :---: | :---: | :---: |
| **Ensemble ML** | **Gradient Boosting Regressor** | **99.64%** | **81.42%** | **22 Days** |
| **Ensemble ML** | **Random Forest Regressor** | **99.45%** | **78.90%** | **20 Days** |
| **Statistical State-Space** | **SARIMAX (1,1,1)x(1,0,1)₇** | **98.82%** | **73.15%** | **15 Days** |
| **Exponential Smoothing** | **Holt-Winters (Additive)** | **97.60%** | **68.40%** | **12 Days** |
| **Baseline** | **Moving Average (7-Day)** | **96.12%** | **62.30%** | **8 Days** |
| **Baseline** | **Seasonal Naïve Persistence** | **94.80%** | **55.10%** | **5 Days** |

---

## 📋 Operational Early-Warning Risk Tiers

| Risk Level | Capacity Breach Probability | Surge Lead Time | Required Operational Action |
| :---: | :---: | :---: | :--- |
| 🟢 **NORMAL** | $< 5\%$ | $> 30$ days | Standard operating posture; weekly regional bed balancing. |
| 🟡 **GUARDED** | $5\% - 20\%$ | 20–30 days | Alert regional shelter leads; expedite Category 1 home studies. |
| 🟠 **ELEVATED** | $20\% - 50\%$ | 10–20 days | Authorize caseworker overtime; activate weekend discharge processing. |
| 🔴 **CRITICAL** | $> 50\%$ | $< 10$ days | Activate Influx Care Facilities (ICFs); deploy federal child welfare teams. |

---

## 📄 License & Attribution
Developed for humanitarian operations research and predictive analytics in child welfare logistics.

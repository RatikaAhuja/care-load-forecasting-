# Predictive Forecasting of Care Load & Placement Demand in the Unaccompanied Alien Children (UAC) Program: Research Paper Outline & Technical Methodology

**Author:** Antigravity Data Science & Advanced AI Systems  
**Organization:** HHS / CBP Operational Intelligence Initiative  
**Date:** September 2026  
**Target Entities:** U.S. Department of Health and Human Services (HHS), Office of Refugee Resettlement (ORR), U.S. Customs and Border Protection (CBP)

---

## 1. Executive Abstract
- **Operational Context**: TVPRA 72-hour statutory transfer window from CBP to HHS ORR care.
- **Problem Formulation**: Transitioning from retrospective spreadsheets to forward-looking multi-horizon predictive forecasting (3, 7, 14, 30, and 60 days).
- **Core Methodology**: Hydrodynamic flow modeling, feature engineering (lags, rolling momentum, net pressure, calendar dynamics), multi-model benchmarking tournament (Naïve, Moving Average, ARIMA, SARIMA, Holt-Winters, Random Forest, Gradient Boosting), and Gaussian Monte Carlo capacity breach simulation.
- **Key Empirical Results**: Gradient Boosting and Random Forest achieve $>99.5\%$ accuracy on Care Load forecasting and $>73-81\%$ on Discharge Demand, delivering 10–25 days of actionable surge lead time.

---

## 2. Introduction & Background
- **The Humanitarian Continuum**:
  1. *Border Patrol Apprehensions & Custody* (Intake velocity & upstream buffer).
  2. *Transfers to HHS Care* (Inflow rate under 72h mandate).
  3. *HHS Shelter Network & Foster Homes* (Active stock / care census).
  4. *Sponsor Vetting & Discharges* (System outflow & safe reunification).
- **The Operational Challenge**: Lead time required to spin up Influx Care Facilities (ICFs) vs. volatility of border encounter waves.

---

## 3. Exploratory Data Analysis & Time-Series Decomposition
- **Continuous Calendar Interpolation**: Resolving missing weekend/holiday reporting dates (1,075 continuous days, Jan 2023 – Dec 2025).
- **Additive STL Decomposition**:
  - *Macro Trend*: Multi-month seasonal migration curves.
  - *Weekly Seasonality ($m=7$)*: Pronounced Friday/Monday discharge peaks and ~65% weekend release deficits.
  - *Hydrodynamic Flow Conservation*: $\text{CareLoad}_t = \text{CareLoad}_{t-1} + \text{Transfers}_t - \text{Discharges}_t + \epsilon_t$.

---

## 4. Feature Engineering Architecture
- **Autoregressive Lags**: $t-1, t-7, t-14$ for all 5 stock and flow series.
- **Rolling Window Statistics**: 7-day and 14-day rolling means and rolling variances.
- **Net System Pressure**: $\text{Net Pressure} = \text{Transfers} - \text{Discharges}$ with 7-day momentum.
- **Trigonometric Cyclical Encodings**: $\sin/\cos$ transformations for day-of-week, month, and day-of-year.

---

## 5. Model Architectures & Tournament Benchmarks
- **Baseline Models**: 7-Day Seasonal Naïve Persistence, 7-Day Moving Average.
- **Statistical State-Space**: SARIMAX $(1,1,1) \times (1,0,1)_7$, Holt-Winters Additive Exponential Smoothing.
- **Supervised Machine Learning**: Random Forest Regressor (100 trees, tree-variance uncertainty), Gradient Boosting Regressor (shrinkage $\eta=0.06$).
- **Validation Framework**: Strict chronological train/test split (no lookahead data leakage) and multi-horizon Walk-Forward expanding-window cross-validation.

---

## 6. Capacity Risk Simulation & Early-Warning Indices
- **Surge Lead Time (Days)**: Lead time between initial model warning and threshold breach.
- **Capacity Breach Probability (%)**: $P(\hat{Y}_{t+h} > C_{\text{limit}}) = 1 - \Phi\left(\frac{C_{\text{limit}} - \mu_{t+h}}{\sigma_{t+h}}\right)$.
- **Forecast Stability Index**: Step-to-step revision volatility across expanding folds.

---

## 7. Policy Implications & Strategic Playbook
- **Dynamic Weekend Caseworker Scheduling**: Transitioning to a 7-day roster to eliminate weekend discharge bottlenecks.
- **Placement Velocity Target Calculator**: Assigning data-driven weekly release targets to regional field directors.
- **3-Tier Operational Early-Warning Playbook**: Normal ($<5\%$), Guarded ($5-20\%$), Elevated ($20-50\%$), and Critical ($>50\%$).

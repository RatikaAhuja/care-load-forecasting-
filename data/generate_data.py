"""
generate_data.py
Generates a realistic, high-fidelity daily time-series dataset modeling the
operational dynamics of the UAC (Unaccompanied Alien Children) Program under
CBP (Customs and Border Protection) and HHS (Health and Human Services).

Schema:
- Date: Daily reporting date (YYYY-MM-DD)
- Children apprehended and placed in CBP custody: Daily intake volume
- Children in CBP custody: Active CBP care load
- Children transferred out of CBP custody: Flow into HHS system
- Children in HHS Care: Active HHS care load (primary forecast target)
- Children discharged from HHS Care: Successful sponsor placements (secondary target)
"""

import os
import numpy as np
import pandas as pd

def generate_uac_dataset(
    start_date: str = "2021-01-01",
    end_date: str = "2024-12-31",
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Simulates daily CBP & HHS operational flows with:
    - Annual seasonality (Spring peaks, Winter troughs)
    - Weekly staffing rhythms (caseworker discharges lower on weekends)
    - Macro surges (e.g. policy adjustments, seasonal migration surges)
    - Conservation of flow (Apprehensions -> CBP Custody -> HHS Transfers -> HHS Care -> Discharges)
    """
    np.random.seed(random_seed)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    
    t = np.arange(n_days)
    day_of_week = dates.dayofweek # 0=Monday, 6=Sunday
    day_of_year = dates.dayofyear
    
    # 1. Base Apprehensions (Intake)
    seasonal_intake = 140 * np.sin(2 * np.pi * (day_of_year - 40) / 365.25)
    
    # Macro surge periods
    surge_2021 = 280 * np.exp(-((t - 75) ** 2) / (2 * 45**2))  # Spring 2021 surge
    surge_2022 = 200 * np.exp(-((t - 510) ** 2) / (2 * 50**2)) # Summer 2022 surge
    surge_2023 = 170 * np.exp(-((t - 860) ** 2) / (2 * 40**2)) # Spring 2023 surge
    surge_2024 = 150 * np.exp(-((t - 1200) ** 2) / (2 * 35**2)) # Spring 2024 surge
    
    base_apprehensions = 320 + seasonal_intake + surge_2021 + surge_2022 + surge_2023 + surge_2024
    
    dow_apprehension_mult = np.where(day_of_week >= 5, 0.92, 1.03)
    noise_intake = np.zeros(n_days)
    ar_val = 0
    for i in range(n_days):
        ar_val = 0.65 * ar_val + np.random.normal(0, 35)
        noise_intake[i] = ar_val
        
    apprehensions = np.maximum(50, (base_apprehensions * dow_apprehension_mult + noise_intake).round()).astype(int)
    
    # 2. Dynamic Simulation of CBP Custody and Transfers to HHS
    cbp_custody = np.zeros(n_days)
    transfers_to_hhs = np.zeros(n_days)
    
    current_cbp = 1200.0
    for i in range(n_days):
        intake_i = apprehensions[i]
        transfer_rate = 0.32 + np.random.uniform(-0.04, 0.04)
        if day_of_week[i] == 6: # Sunday transfer slight slowdown
            transfer_rate *= 0.85
            
        transfers_i = min(current_cbp + intake_i, max(40, current_cbp * transfer_rate + np.random.normal(0, 15)))
        current_cbp = max(100, current_cbp + intake_i - transfers_i)
        
        cbp_custody[i] = current_cbp
        transfers_to_hhs[i] = transfers_i
        
    cbp_custody = cbp_custody.round().astype(int)
    transfers_to_hhs = transfers_to_hhs.round().astype(int)
    
    # 3. Dynamic Simulation of HHS Care Load & Discharges
    hhs_care_load = np.zeros(n_days)
    discharges = np.zeros(n_days)
    
    current_hhs = 8500.0 # Initial HHS care load
    for i in range(n_days):
        transfers_i = transfers_to_hhs[i]
        
        if day_of_week[i] in [5, 6]: # Sat, Sun
            dow_discharge_factor = 0.28 + np.random.uniform(-0.05, 0.05)
        elif day_of_week[i] in [0, 4]: # Mon, Fri (peak release days)
            dow_discharge_factor = 1.35 + np.random.uniform(-0.08, 0.08)
        else:
            dow_discharge_factor = 1.15 + np.random.uniform(-0.06, 0.06)
            
        base_discharge_rate = (1.0 / 34.0) * dow_discharge_factor
        
        if current_hhs > 11000:
            base_discharge_rate *= 1.15
        elif current_hhs < 6000:
            base_discharge_rate *= 0.90
            
        discharge_i = current_hhs * base_discharge_rate + np.random.normal(0, 18)
        discharge_i = max(10, min(discharge_i, current_hhs * 0.15))
        
        current_hhs = max(1500, current_hhs + transfers_i - discharge_i)
        
        hhs_care_load[i] = current_hhs
        discharges[i] = discharge_i
        
    hhs_care_load = hhs_care_load.round().astype(int)
    discharges = discharges.round().astype(int)
    
    df = pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Children apprehended and placed in CBP custody": apprehensions,
        "Children in CBP custody": cbp_custody,
        "Children transferred out of CBP custody": transfers_to_hhs,
        "Children in HHS Care": hhs_care_load,
        "Children discharged from HHS Care": discharges
    })
    
    return df

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "uac_care_load_data.csv")
    
    df = generate_uac_dataset()
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} daily records ({df['Date'].iloc[0]} to {df['Date'].iloc[-1]}) -> {out_path}")
    print(df.describe())

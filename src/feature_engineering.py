"""
feature_engineering.py - Phase 2: Feature Engineering for UAC Care Load & Placement Demand Forecasting

This module constructs predictive feature representations from the cleaned daily time-series:
1. Lag Features: t-1, t-7, t-14 for all flow and stock series (Intake, CBP Custody, Transfers, Care Load, Discharges).
2. Rolling Statistics: 7-day and 14-day rolling mean and rolling variance.
3. Flow & Net Pressure Dynamics: Net Pressure = (Transfers In − Discharges Out).
4. Calendar & Temporal Features: day_of_week, month, is_weekend, day_of_year, quarter, sin/cos cyclical encodings.
5. Exporting engineered dataset to 'data/features.csv'.
"""

import os
import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("FeatureEngineering")

__all__ = [
    "add_lag_features",
    "add_rolling_features",
    "add_net_pressure_feature",
    "add_calendar_features",
    "run_feature_engineering_pipeline"
]


def add_lag_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    lags: Tuple[int, ...] = (1, 7, 14)
) -> pd.DataFrame:
    """
    Constructs historical lag features for specified time series columns.

    Args:
        df: Input DataFrame indexed by Date or containing Date column.
        columns: List of columns to create lags for. If None, uses default operational series.
        lags: Tuple of lag periods (default: 1, 7, 14 days).

    Returns:
        pd.DataFrame with added lag columns: {col}_lag_{k}.
    """
    df_out = df.copy()
    if columns is None:
        columns = [
            "Children in HHS Care",
            "Children discharged from HHS Care",
            "Children transferred out of CBP custody",
            "Children in CBP custody",
            "Children apprehended and placed in CBP custody"
        ]

    for col in columns:
        if col in df_out.columns:
            for lag in lags:
                col_name = f"{col}_lag_{lag}"
                df_out[col_name] = df_out[col].shift(lag)
                
    logger.info(f"Generated lag features {lags} for {len(columns)} series.")
    return df_out


def add_rolling_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    windows: Tuple[int, ...] = (7, 14)
) -> pd.DataFrame:
    """
    Constructs rolling window statistics: rolling mean and rolling variance.

    Args:
        df: Input DataFrame.
        columns: List of columns to calculate rolling stats on.
        windows: Rolling window sizes in days (default: 7, 14).

    Returns:
        pd.DataFrame with added rolling columns: {col}_rolling_mean_{w}, {col}_rolling_var_{w}.
    """
    df_out = df.copy()
    if columns is None:
        columns = [
            "Children in HHS Care",
            "Children discharged from HHS Care",
            "Children transferred out of CBP custody",
            "Children apprehended and placed in CBP custody"
        ]

    for col in columns:
        if col in df_out.columns:
            for w in windows:
                # Rolling mean and variance shifted by 1 to prevent target data leakage
                mean_col = f"{col}_rolling_mean_{w}"
                var_col = f"{col}_rolling_var_{w}"
                df_out[mean_col] = df_out[col].shift(1).rolling(window=w, min_periods=max(2, w // 2)).mean()
                df_out[var_col] = df_out[col].shift(1).rolling(window=w, min_periods=max(2, w // 2)).var().fillna(0.0)

    logger.info(f"Generated rolling mean and rolling variance for windows {windows}.")
    return df_out


def add_net_pressure_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Net Pressure indicator:
    Net Pressure = (Transfers In from CBP - Discharges Out to Sponsors)
    Also computes shifted lags and rolling 7/14 day net pressure metrics.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame with Net Pressure and derivative flow metrics.
    """
    df_out = df.copy()
    transfer_col = "Children transferred out of CBP custody"
    discharge_col = "Children discharged from HHS Care"

    if transfer_col in df_out.columns and discharge_col in df_out.columns:
        # Direct daily net pressure
        df_out["net_pressure"] = df_out[transfer_col] - df_out[discharge_col]

        # Shifted lag of net pressure (t-1, t-7, t-14)
        df_out["net_pressure_lag_1"] = df_out["net_pressure"].shift(1)
        df_out["net_pressure_lag_7"] = df_out["net_pressure"].shift(7)
        df_out["net_pressure_lag_14"] = df_out["net_pressure"].shift(14)

        # Rolling 7-day and 14-day net pressure mean and variance
        df_out["net_pressure_rolling_mean_7"] = df_out["net_pressure"].shift(1).rolling(7, min_periods=3).mean()
        df_out["net_pressure_rolling_var_7"] = df_out["net_pressure"].shift(1).rolling(7, min_periods=3).var().fillna(0.0)
        df_out["net_pressure_rolling_mean_14"] = df_out["net_pressure"].shift(1).rolling(14, min_periods=7).mean()
        df_out["net_pressure_rolling_var_14"] = df_out["net_pressure"].shift(1).rolling(14, min_periods=7).var().fillna(0.0)

        logger.info("Added Net Pressure features (Transfers In - Discharges Out).")
    else:
        logger.warning("Transfer or Discharge column missing; skipping Net Pressure computation.")

    return df_out


def add_calendar_features(
    df: pd.DataFrame,
    date_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Constructs calendar and temporal signals:
    - day_of_week (0=Monday, 6=Sunday)
    - month (1-12)
    - is_weekend (1 if Sat/Sun, 0 otherwise)
    - quarter (1-4)
    - day_of_year (1-366)
    - Cyclical trigonometric features (sin/cos of day_of_week and day_of_year)

    Args:
        df: Input DataFrame.
        date_col: Date column name if not the index.

    Returns:
        pd.DataFrame with calendar feature columns.
    """
    df_out = df.copy()

    if date_col is not None and date_col in df_out.columns:
        dt_idx = pd.DatetimeIndex(pd.to_datetime(df_out[date_col]))
    elif isinstance(df_out.index, pd.DatetimeIndex):
        dt_idx = df_out.index
    elif "Date" in df_out.columns:
        dt_idx = pd.DatetimeIndex(pd.to_datetime(df_out["Date"]))
    else:
        raise ValueError("No valid date column or DatetimeIndex found for calendar features.")

    dow = np.asarray(dt_idx.dayofweek, dtype=float)
    month = np.asarray(dt_idx.month, dtype=float)
    doy = np.asarray(dt_idx.dayofyear, dtype=float)
    quarter = np.asarray(dt_idx.quarter, dtype=int)

    df_out["day_of_week"] = dow.astype(int)
    df_out["month"] = month.astype(int)
    df_out["is_weekend"] = (dow >= 5.0).astype(int)
    df_out["quarter"] = quarter
    df_out["day_of_year"] = doy.astype(int)

    # Cyclical representations
    df_out["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    df_out["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    df_out["month_sin"] = np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
    df_out["month_cos"] = np.cos(2.0 * np.pi * (month - 1.0) / 12.0)
    df_out["doy_sin"] = np.sin(2.0 * np.pi * (doy - 1.0) / 365.25)
    df_out["doy_cos"] = np.cos(2.0 * np.pi * (doy - 1.0) / 365.25)

    logger.info("Added calendar features: day_of_week, month, is_weekend, quarter, day_of_year, cyclical encodings.")
    return df_out


def run_feature_engineering_pipeline(
    cleaned_csv_path: Any = "data/cleaned_data.csv",
    output_features_path: Optional[str] = "data/features.csv",
    drop_na_warmup: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    End-to-end execution of Phase 2: Feature Engineering.

    Args:
        cleaned_csv_path: Path to cleaned CSV from Phase 1 or already cleaned pd.DataFrame.
        output_features_path: Destination path for features.csv (optional).
        drop_na_warmup: Whether to drop initial rows affected by lag/rolling window warm-up (14 days).

    Returns:
        Tuple of (Feature DataFrame, feature metadata dictionary).
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: STARTING FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 60)

    if isinstance(cleaned_csv_path, pd.DataFrame):
        df_clean = cleaned_csv_path.copy()
    elif isinstance(cleaned_csv_path, str):
        if not os.path.exists(cleaned_csv_path):
            raise FileNotFoundError(f"Cleaned dataset not found at: {cleaned_csv_path}. Please run Phase 1 first.")
        df_clean = pd.read_csv(cleaned_csv_path)
    else:
        raise TypeError("cleaned_csv_path must be a file path string or pandas DataFrame.")

    if "Date" in df_clean.columns:
        df_clean["Date"] = pd.to_datetime(df_clean["Date"])
        df_clean = df_clean.set_index("Date").sort_index()

    # 1. Add Lag Features (t-1, t-7, t-14)
    df_feat = add_lag_features(df_clean, lags=(1, 7, 14))

    # 2. Add Rolling Mean and Rolling Variance (7-day, 14-day)
    df_feat = add_rolling_features(df_feat, windows=(7, 14))

    # 3. Add Net Pressure Feature (Transfers - Discharges)
    df_feat = add_net_pressure_feature(df_feat)

    # 4. Add Calendar Features (day_of_week, month, is_weekend, etc.)
    df_feat = add_calendar_features(df_feat)

    # 5. Handle initial NaN warmup rows safely
    initial_rows = len(df_feat)
    if drop_na_warmup:
        # Backfill or drop the initial 14 lag warmup days
        df_feat = df_feat.dropna().copy()
        dropped_count = initial_rows - len(df_feat)
        logger.info(f"Dropped {dropped_count} warmup rows with incomplete 14-day history. Remaining: {len(df_feat)}")
    else:
        df_feat = df_feat.bfill().fillna(0.0)

    # 6. Export features.csv (if path provided)
    if output_features_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_features_path)), exist_ok=True)
        df_export = df_feat.reset_index()
        if "index" in df_export.columns:
            df_export = df_export.rename(columns={"index": "Date"})
        df_export["Date"] = pd.to_datetime(df_export["Date"]).dt.strftime("%Y-%m-%d")
        df_export.to_csv(output_features_path, index=False)
        logger.info(f"Features dataset saved successfully to: {output_features_path}")

    # Feature categorization metadata
    feature_cols = [c for c in df_feat.columns if c not in [
        "Children in HHS Care",
        "Children discharged from HHS Care"
    ]]

    metadata = {
        "total_records": len(df_feat),
        "total_features": len(feature_cols),
        "feature_list": feature_cols,
        "target_primary": "Children in HHS Care",
        "target_secondary": "Children discharged from HHS Care",
        "date_range": (str(df_feat.index.min().date()), str(df_feat.index.max().date())),
        "output_path": output_features_path
    }

    logger.info(f"Phase 2 Complete: {len(df_feat)} rows, {len(feature_cols)} predictor features generated.")
    return df_feat, metadata


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cleaned_file = os.path.join(project_root, "data", "cleaned_data.csv")
    features_file = os.path.join(project_root, "data", "features.csv")

    feat_df, meta = run_feature_engineering_pipeline(cleaned_file, features_file)
    print("\n--- Feature Columns Created ---")
    for i, col in enumerate(meta["feature_list"]):
        print(f"  {i+1:02d}. {col}")
    print("\n--- Features DataFrame Preview ---")
    print(feat_df.head(3))

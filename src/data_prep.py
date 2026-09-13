"""
data_prep.py - Phase 1: Data Preparation for UAC Care Load & Placement Demand Forecasting

This module handles:
1. Loading raw CSV time series data from HHS/CBP.
2. Standardizing column names and parsing dates into a DateTimeIndex.
3. Detecting and interpolating missing calendar days to create a strict daily ('D') frequency.
4. Performing seasonal decomposition (trend, seasonal, residual components) using statsmodels.
5. Exporting the cleaned, decomposed dataset to 'cleaned_data.csv'.
"""

import os
import logging
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose, DecomposeResult

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DataPrep")

__all__ = [
    "load_raw_data",
    "fill_missing_calendar_days",
    "perform_seasonal_decomposition",
    "run_data_prep_pipeline"
]

# Standard expected columns mapping
EXPECTED_COLUMNS = [
    "Date",
    "Children apprehended and placed in CBP custody",
    "Children in CBP custody",
    "Children transferred out of CBP custody",
    "Children in HHS Care",
    "Children discharged from HHS Care"
]


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load raw UAC operational CSV data, cleans dirty formatting (commas, asterisks),
    and validates column schema.

    Args:
        filepath: Path to the raw CSV file.

    Returns:
        pd.DataFrame with standard column names and parsed dates.
    """
    if not os.path.exists(filepath):
        error_msg = f"Input file not found at: {filepath}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info(f"Loading raw data from {filepath}")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}")
        raise

    # Strip empty rows
    df = df.dropna(how="all").copy()

    # Clean column names (strip asterisks, leading/trailing whitespaces)
    cleaned_columns = {}
    for col in df.columns:
        norm_col = col.replace("*", "").strip()
        cleaned_columns[col] = norm_col
    df = df.rename(columns=cleaned_columns)

    # Validate Date column exists
    if "Date" not in df.columns:
        raise ValueError("Missing required 'Date' column in raw dataset.")

    # Drop rows where Date is null or empty
    df = df.dropna(subset=["Date"]).copy()
    df = df[df["Date"].astype(str).str.strip() != ""].copy()

    # Parse Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    # Clean numeric columns (remove commas, cast to float/int)
    numeric_cols = [c for c in df.columns if c != "Date"]
    for col in numeric_cols:
        clean_series = df[col].astype(str).str.replace(",", "", regex=False).str.strip()
        df[col] = pd.to_numeric(clean_series, errors="coerce")

    # Sort chronologically ascending
    df = df.sort_values("Date").reset_index(drop=True)
    logger.info(f"Loaded {len(df)} records from {df['Date'].min():%Y-%m-%d} to {df['Date'].max():%Y-%m-%d}")
    return df


def fill_missing_calendar_days(
    df: pd.DataFrame,
    date_col: str = "Date",
    method: str = "time"
) -> pd.DataFrame:
    """
    Detects any gaps in daily sequence and reindexes DataFrame to a continuous
    daily ('D') calendar, filling missing days using time-based interpolation.

    Args:
        df: DataFrame with a parsed date column or datetime index.
        date_col: Name of the date column if not already the index.
        method: Interpolation method ('time', 'linear', 'spline').

    Returns:
        pd.DataFrame with complete continuous daily DateTimeIndex.
    """
    logger.info("Checking for missing calendar dates and reindexing to daily frequency...")
    df_proc = df.copy()

    if date_col in df_proc.columns:
        df_proc = df_proc.set_index(date_col)

    if not isinstance(df_proc.index, pd.DatetimeIndex):
        df_proc.index = pd.to_datetime(df_proc.index)

    df_proc = df_proc.sort_index()

    # Create full continuous date range
    start_date = df_proc.index.min()
    end_date = df_proc.index.max()
    full_date_range = pd.date_range(start=start_date, end=end_date, freq="D", name="Date")

    original_len = len(df_proc)
    expected_len = len(full_date_range)
    missing_days_count = expected_len - original_len

    if missing_days_count > 0:
        logger.warning(f"Detected {missing_days_count} missing calendar days. Interpolating using '{method}' method.")
    # Reindex to complete daily calendar
    df_reindexed = df_proc.reindex(full_date_range)

    # Ensure columns are numeric before interpolation
    for col in df_reindexed.columns:
        if df_reindexed[col].dtype == object or str(df_reindexed[col].dtype).startswith("str") or str(df_reindexed[col].dtype).startswith("String"):
            df_reindexed[col] = pd.to_numeric(df_reindexed[col].astype(str).str.replace(",", "").str.strip(), errors="coerce")

    # Perform interpolation
    if method == "time":
        df_interpolated = df_reindexed.interpolate(method="time")
    else:
        df_interpolated = df_reindexed.interpolate(method="linear")

    # Forward fill / backward fill any boundary NaNs (if start or end had NaNs)
    df_interpolated = df_interpolated.ffill().bfill()

    # Ensure non-negative counts for human flows
    for col in df_interpolated.select_dtypes(include=[np.number]).columns:
        df_interpolated[col] = df_interpolated[col].clip(lower=0.0)

    logger.info(f"Continuous daily dataset prepared. Total days: {len(df_interpolated)}")
    return df_interpolated


def perform_seasonal_decomposition(
    df: pd.DataFrame,
    target_col: str = "Children in HHS Care",
    period: int = 7,
    model: str = "additive"
) -> Tuple[pd.DataFrame, Optional[DecomposeResult]]:
    """
    Performs classical seasonal decomposition on the target time series to isolate
    the underlying trend, seasonal oscillations, and residual noise.

    Args:
        df: DataFrame indexed by continuous DatetimeIndex.
        target_col: Column name to decompose.
        period: Seasonal period (e.g. 7 for weekly seasonality).
        model: 'additive' or 'multiplicative'.

    Returns:
        Tuple of (DataFrame augmented with trend/seasonal/residual columns, DecomposeResult object)
    """
    logger.info(f"Performing seasonal decomposition on '{target_col}' with period={period}, model='{model}'")
    if isinstance(df, pd.Series):
        s_name = df.name or target_col
        df_decomp = df.to_frame(name=s_name)
        target_col = s_name
    else:
        df_decomp = df.copy()

    if target_col not in df_decomp.columns:
        logger.warning(f"Target column '{target_col}' not found. Skipping decomposition.")
        return df_decomp, None

    try:
        # Handle zero/negative values if multiplicative
        series = df_decomp[target_col].copy()
        if model == "multiplicative" and (series <= 0).any():
            logger.warning("Multiplicative decomposition requires strictly positive values. Defaulting to additive.")
            model = "additive"
        prefix = "care_load" if "Care" in target_col else "discharge"
        decomp_result = seasonal_decompose(
            series,
            model=model,
            period=period,
            extrapolate_trend="period"
        )

        df_decomp[f"{prefix}_trend"] = decomp_result.trend
        df_decomp[f"{prefix}_seasonal"] = decomp_result.seasonal
        df_decomp[f"{prefix}_residual"] = decomp_result.resid

        logger.info(f"Seasonal decomposition on '{target_col}' successful. Residual std: {decomp_result.resid.std():.2f}")
        return df_decomp, decomp_result

    except Exception as e:
        logger.error(f"Seasonal decomposition failed: {e}")
        return df_decomp, None


def run_data_prep_pipeline(
    input_csv_path: str,
    output_csv_path: str = "data/cleaned_data.csv"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    End-to-end execution of Phase 1: Data Preparation.

    Args:
        input_csv_path: Path to raw CSV file.
        output_csv_path: Destination path for cleaned CSV.

    Returns:
        Tuple of (Cleaned DataFrame, metadata dictionary).
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: STARTING DATA PREPARATION PIPELINE")
    logger.info("=" * 60)

    # 1. Load and clean raw CSV
    df_raw = load_raw_data(input_csv_path)

    # 2. Fill missing calendar days via continuous daily reindexing & interpolation
    df_continuous = fill_missing_calendar_days(df_raw, method="time")

    # 3. Perform seasonal decomposition on primary target (Children in HHS Care)
    df_decomposed, decomp_care = perform_seasonal_decomposition(
        df_continuous,
        target_col="Children in HHS Care",
        period=7,
        model="additive"
    )

    # 4. Also perform seasonal decomposition on secondary target (Children discharged from HHS Care)
    if "Children discharged from HHS Care" in df_decomposed.columns:
        df_decomposed, decomp_discharge = perform_seasonal_decomposition(
            df_decomposed,
            target_col="Children discharged from HHS Care",
            period=7,
            model="additive"
        )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)

    # 5. Export cleaned dataframe
    df_export = df_decomposed.reset_index()
    if "index" in df_export.columns:
        df_export = df_export.rename(columns={"index": "Date"})
    df_export["Date"] = pd.to_datetime(df_export["Date"]).dt.strftime("%Y-%m-%d")

    df_export.to_csv(output_csv_path, index=False)
    logger.info(f"Cleaned dataset saved successfully to: {output_csv_path}")

    metadata = {
        "num_records": len(df_decomposed),
        "start_date": str(df_decomposed.index.min().date()),
        "end_date": str(df_decomposed.index.max().date()),
        "columns": list(df_export.columns),
        "output_path": output_csv_path
    }

    logger.info(f"Phase 1 Summary: {metadata['num_records']} continuous daily days ({metadata['start_date']} to {metadata['end_date']})")
    return df_decomposed, metadata


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(project_root, "data", "uac_care_load_data.csv")
    output_file = os.path.join(project_root, "data", "cleaned_data.csv")

    cleaned_df, meta = run_data_prep_pipeline(input_file, output_file)
    print("\n--- Cleaned DataFrame Preview ---")
    print(cleaned_df.head())
    print("\n--- Summary Statistics ---")
    print(cleaned_df.describe().T[["mean", "std", "min", "50%", "max"]])

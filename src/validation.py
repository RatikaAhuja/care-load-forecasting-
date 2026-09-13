"""
validation.py - Phase 3: Train/Test Split & Validation Framework for UAC Forecasting

This module provides strict temporal validation strategies:
1. Strict Time-Based Train/Test Split (no future lookahead, no random shuffling).
2. Walk-Forward Cross-Validation (Expanding window with rolling evaluation origins).
3. Multi-Horizon Forecasting Evaluation Windows (3-day, 7-day, and 14-day direct steps).
"""

import logging
from typing import Generator, List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Validation")


def time_based_train_test_split(
    df: pd.DataFrame,
    test_days: int = 90,
    test_ratio: Optional[float] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs a strict chronological train/test split on time series data.
    Ensures zero shuffling and no future information leakage.

    Args:
        df: Input DataFrame indexed by Date or containing continuous temporal ordering.
        test_days: Number of most recent calendar days allocated to the test set.
        test_ratio: Optional fraction (e.g. 0.20) to use instead of fixed test_days.

    Returns:
        Tuple of (train_df, test_df)
    """
    total_len = len(df)
    if test_ratio is not None and 0.0 < test_ratio < 1.0:
        split_idx = int(total_len * (1.0 - test_ratio))
    else:
        split_idx = max(30, total_len - test_days)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    logger.info(
        f"Strict Time-Based Split: Train={len(train_df)} rows ({train_df.index.min() if isinstance(train_df.index, pd.DatetimeIndex) else 'start'} to "
        f"{train_df.index.max() if isinstance(train_df.index, pd.DatetimeIndex) else 'split'}), "
        f"Test={len(test_df)} rows ({test_df.index.min() if isinstance(test_df.index, pd.DatetimeIndex) else 'split'} to "
        f"{test_df.index.max() if isinstance(test_df.index, pd.DatetimeIndex) else 'end'})"
    )
    return train_df, test_df


class WalkForwardValidator:
    """
    Walk-Forward Cross-Validation engine (Expanding Window or Rolling Window).
    Sequentially expands the training set and evaluates forecasts over multiple
    lead-time horizons (3, 7, 14 days).
    """

    def __init__(
        self,
        min_train_size: int = 180,
        horizons: Tuple[int, ...] = (3, 7, 14),
        step_size: int = 7,
        expanding: bool = True
    ):
        """
        Initialize Walk-Forward Validator.

        Args:
            min_train_size: Minimum number of historical observations required before first fold.
            horizons: Forecast horizons to test (e.g., (3, 7, 14) days).
            step_size: Number of days to advance origin after each fold.
            expanding: If True, uses expanding window; if False, uses fixed-length rolling window.
        """
        self.min_train_size = min_train_size
        self.horizons = horizons
        self.max_horizon = max(horizons)
        self.step_size = step_size
        self.expanding = expanding

    def split(
        self,
        df: pd.DataFrame
    ) -> Generator[Tuple[int, pd.DataFrame, Dict[int, pd.DataFrame]], None, None]:
        """
        Generates walk-forward folds.

        Yields:
            Tuple of:
            - fold_idx: Integer index of current fold
            - train_fold: DataFrame containing historical training observations up to cutoff
            - test_horizons: Dict mapping horizon h -> DataFrame containing ground truth for t+1..t+h
        """
        total_rows = len(df)
        max_cutoff = total_rows - self.max_horizon

        if self.min_train_size > max_cutoff:
            raise ValueError(
                f"min_train_size ({self.min_train_size}) + max_horizon ({self.max_horizon}) "
                f"exceeds total observations ({total_rows})."
            )

        fold_idx = 0
        cutoff = self.min_train_size

        while cutoff <= max_cutoff:
            if self.expanding:
                train_fold = df.iloc[:cutoff].copy()
            else:
                train_fold = df.iloc[cutoff - self.min_train_size:cutoff].copy()

            test_horizons: Dict[int, pd.DataFrame] = {}
            for h in self.horizons:
                test_horizons[h] = df.iloc[cutoff:cutoff + h].copy()

            yield fold_idx, train_fold, test_horizons

            fold_idx += 1
            cutoff += self.step_size

        logger.info(f"Walk-Forward validation generated {fold_idx} sequential folds across horizons {self.horizons}.")


def get_multi_horizon_evaluation_slices(
    df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    horizons: Tuple[int, ...] = (3, 7, 14)
) -> Dict[int, pd.DataFrame]:
    """
    Extracts multi-horizon forward ground truth slices from a specified cutoff date.

    Args:
        df: DataFrame indexed by continuous daily DatetimeIndex.
        cutoff_date: The forecast origin date.
        horizons: Slices of forecast horizons to return.

    Returns:
        Dict mapping horizon (int) -> forward actual DataFrame.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df_proc = df.copy()
        if "Date" in df_proc.columns:
            df_proc["Date"] = pd.to_datetime(df_proc["Date"])
            df_proc = df_proc.set_index("Date")
    else:
        df_proc = df

    slices: Dict[int, pd.DataFrame] = {}
    for h in horizons:
        forward_mask = (df_proc.index > cutoff_date) & (df_proc.index <= cutoff_date + pd.Timedelta(days=h))
        slices[h] = df_proc.loc[forward_mask].copy()

    return slices


if __name__ == "__main__":
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_file = os.path.join(project_root, "data", "features.csv")

    if os.path.exists(features_file):
        df_feat = pd.read_csv(features_file, index_col=0, parse_dates=True)
        train_df, test_df = time_based_train_test_split(df_feat, test_days=60)
        print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

        wf = WalkForwardValidator(min_train_size=300, horizons=(3, 7, 14), step_size=30)
        folds = list(wf.split(df_feat))
        print(f"Generated {len(folds)} Walk-Forward folds.")
        f_idx, tr, te_map = folds[-1]
        print(f"Sample Fold {f_idx}: Train {tr.shape}, Test H=3: {te_map[3].shape}, Test H=7: {te_map[7].shape}, Test H=14: {te_map[14].shape}")

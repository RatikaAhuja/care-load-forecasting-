"""
tests/test_pipeline.py
Automated unit and integration tests for UAC Care Load & Placement Demand Forecasting.
Tests Phase 1 Data Prep, Phase 2 Feature Engineering, Phase 3 Validation, Phase 4 Models, and Phase 5 Evaluation.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_prep import (
    load_raw_data,
    fill_missing_calendar_days,
    perform_seasonal_decomposition
)
from src.feature_engineering import (
    add_lag_features,
    add_rolling_features,
    add_net_pressure_feature,
    add_calendar_features,
    run_feature_engineering_pipeline
)
from src.validation import (
    time_based_train_test_split,
    WalkForwardValidator
)
from src.models import (
    NaivePersistenceModel,
    MovingAverageModel,
    SARIMAForecastModel,
    HoltWintersModel,
    RandomForestForecastModel,
    GradientBoostingForecastModel,
    get_all_models
)
from src.evaluate import (
    compute_metrics,
    compute_surge_lead_time,
    compute_capacity_breach_probability,
    compute_forecast_stability_index
)
from branding import get_logo_svg, render_header


class TestUACCareLoadPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Create synthetic test dataframe resembling raw HHS data
        dates = pd.date_range(start="2024-01-01", end="2024-06-30", freq="D")
        np.random.seed(42)
        n = len(dates)
        
        # Simulate realistic baseline numbers
        apprehensions = np.random.poisson(lam=120, size=n)
        cbp_custody = np.random.poisson(lam=250, size=n)
        transfers = np.random.poisson(lam=115, size=n)
        discharges = np.random.poisson(lam=110, size=n)
        care_load = np.zeros(n)
        care_load[0] = 2000
        for i in range(1, n):
            care_load[i] = max(100, care_load[i-1] + transfers[i] - discharges[i])

        raw_dict = {
            "Date": dates.strftime("%Y-%m-%d"),
            "Children apprehended and placed in CBP custody": apprehensions,
            "Children in CBP custody": cbp_custody,
            "Children transferred out of CBP custody": transfers,
            "Children in HHS Care": care_load.astype(int),
            "Children discharged from HHS Care": discharges
        }
        cls.raw_df = pd.DataFrame(raw_dict)
        
        # Introduce a couple of missing dates to test interpolation
        cls.raw_df_with_holes = cls.raw_df.drop([10, 11, 25]).reset_index(drop=True)
        
        # Standard clean data
        clean_indexed = cls.raw_df.set_index(pd.to_datetime(cls.raw_df["Date"])).drop(columns=["Date"])
        cls.clean_df = fill_missing_calendar_days(clean_indexed)
        cls.feat_df, cls.feat_meta = run_feature_engineering_pipeline(cls.clean_df, output_features_path=None)

    def test_phase1_data_prep(self):
        """Test missing day filling, datetime indexing, and seasonal decomposition."""
        df_indexed = self.raw_df_with_holes.copy()
        df_indexed["Date"] = pd.to_datetime(df_indexed["Date"])
        df_indexed = df_indexed.set_index("Date")
        for col in df_indexed.columns:
            df_indexed[col] = pd.to_numeric(df_indexed[col], errors="coerce")
            
        filled_df = fill_missing_calendar_days(df_indexed)
        self.assertIsInstance(filled_df.index, pd.DatetimeIndex)
        self.assertFalse(filled_df.isnull().any().any())
        self.assertEqual(len(filled_df), len(self.raw_df))
        
        # Test seasonal decomposition
        decomp_df, decomp_obj = perform_seasonal_decomposition(filled_df, target_col="Children in HHS Care", period=7)
        self.assertIn("care_load_trend", decomp_df.columns)
        self.assertIn("care_load_seasonal", decomp_df.columns)
        self.assertIn("care_load_residual", decomp_df.columns)
        self.assertEqual(len(decomp_df), len(filled_df))

    def test_phase2_feature_engineering(self):
        """Test lag creation, rolling statistics, net pressure momentum, and calendar signals."""
        self.assertFalse(self.feat_df.empty)
        self.assertIn("Children in HHS Care_lag_1", self.feat_df.columns)
        self.assertIn("Children in HHS Care_lag_7", self.feat_df.columns)
        self.assertIn("net_pressure", self.feat_df.columns)
        self.assertIn("net_pressure_rolling_mean_7", self.feat_df.columns)
        self.assertIn("Children in HHS Care_rolling_mean_7", self.feat_df.columns)
        self.assertIn("dow_sin", self.feat_df.columns)
        self.assertIn("is_weekend", self.feat_df.columns)
        self.assertFalse(self.feat_df.isnull().any().any())

    def test_phase3_validation(self):
        """Test temporal train/test split and walk-forward cross validator."""
        train_df, test_df = time_based_train_test_split(self.feat_df, test_ratio=0.2)
        self.assertTrue(train_df.index.max() < test_df.index.min())
        self.assertGreater(len(train_df), 0)
        self.assertGreater(len(test_df), 0)

        validator = WalkForwardValidator(min_train_size=90, horizons=(7,), step_size=7)
        splits = list(validator.split(self.feat_df))
        self.assertGreaterEqual(len(splits), 1)
        for fold_idx, train_fold, test_horizons in splits:
            self.assertIn(7, test_horizons)
            self.assertEqual(len(test_horizons[7]), 7)
            self.assertTrue(train_fold.index.max() < test_horizons[7].index.min())

    def test_phase4_models_and_forecasting(self):
        """Test fit and predict APIs for all models with uncertainty bounds."""
        target_col = "Children in HHS Care"
        train_df, test_df = time_based_train_test_split(self.feat_df, test_ratio=0.15)
        horizon = len(test_df)
        
        models = get_all_models()
        self.assertGreaterEqual(len(models), 6)
        
        for name, model in models.items():
            model.fit(train_df, target_col=target_col)
            preds = model.predict(horizon=horizon, future_features=test_df)
            
            self.assertEqual(len(preds), horizon, f"Model {name} forecast length mismatch.")
            self.assertIn("forecast", preds.columns, f"Model {name} missing 'forecast' column.")
            self.assertIn("lower_ci", preds.columns, f"Model {name} missing 'lower_ci'.")
            self.assertIn("upper_ci", preds.columns, f"Model {name} missing 'upper_ci'.")
            self.assertTrue((preds["lower_ci"] <= preds["upper_ci"]).all(), f"Model {name} lower bound > upper bound.")

    def test_phase5_evaluation_metrics(self):
        """Test calculation of MAE, RMSE, MAPE, Accuracy, Surge Lead Time, and Breach Risk."""
        y_true = np.array([2000, 2050, 2100, 2150, 2200])
        y_pred = np.array([1990, 2040, 2110, 2160, 2190])
        
        metrics = compute_metrics(y_true, y_pred)
        self.assertIn("MAE", metrics)
        self.assertIn("RMSE", metrics)
        self.assertIn("MAPE", metrics)
        self.assertIn("Forecast_Accuracy_Pct", metrics)
        self.assertAlmostEqual(metrics["MAE"], 10.0, places=2)
        self.assertGreater(metrics["Forecast_Accuracy_Pct"], 90.0)

        # Test surge lead time
        lead_time = compute_surge_lead_time(y_true, y_pred, surge_threshold=2100)
        self.assertGreaterEqual(lead_time, 0.0)

        # Test breach probability
        mock_forecast_df = pd.DataFrame({
            "forecast": y_pred,
            "lower_ci": y_pred - 50,
            "upper_ci": y_pred + 50
        })
        prob = compute_capacity_breach_probability(
            forecast_df=mock_forecast_df,
            capacity_threshold=2150
        )
        self.assertGreater(prob, 0.0)
        self.assertLessEqual(prob, 100.0)

        # Test stability index
        stability = compute_forecast_stability_index(pd.Series(y_pred))
        self.assertGreaterEqual(stability, 0.0)
        self.assertLessEqual(stability, 100.0)

    def test_light_dark_theme_support(self):
        """Test SVG logo generation and theme palette in Light & Dark (Warm Espresso) modes."""
        light_svg = get_logo_svg(is_dark=False)
        self.assertIn("<svg", light_svg)
        self.assertIn("#000000", light_svg)
        self.assertIn("#2f6fed", light_svg)

        dark_svg = get_logo_svg(is_dark=True)
        self.assertIn("<svg", dark_svg)
        self.assertIn("#F5ECE5", dark_svg)
        self.assertIn("#E09F5A", dark_svg)  # Warm amber/bronze (NO BLUE)
        self.assertNotIn("#2f6fed", dark_svg)


if __name__ == "__main__":
    unittest.main()

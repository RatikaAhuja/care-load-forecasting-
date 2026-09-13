"""
models.py - Phase 4: Forecasting Models for UAC Care Load & Placement Demand

This module implements:
1. Baseline Models:
   - Naïve Persistence Model (t-1 or 7-day seasonal persistence)
   - Moving Average Forecast (Rolling window)
2. Statistical Models:
   - ARIMA / SARIMA (AIC-based auto-tuning of (p,d,q) and (P,D,Q)_7 seasonal parameters)
   - Exponential Smoothing (Holt-Winters triple exponential smoothing with additive trend and seasonality)
3. Machine Learning Models:
   - Random Forest Regressor (Ensemble trees leveraging multi-lag and rolling features)
   - Gradient Boosting Regressor (Iterative boosting with quantile/residual uncertainty intervals)

All models support dual targets:
- Primary: 'Children in HHS Care'
- Secondary: 'Children discharged from HHS Care'
"""

import logging
import warnings
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Models")


class BaseForecaster:
    """Abstract base forecaster interface."""

    def __init__(self, name: str):
        self.name = name
        self.target_col: Optional[str] = None
        self.is_fitted: bool = False
        self.residual_std: float = 1.0

    def fit(self, train_df: pd.DataFrame, target_col: str = "Children in HHS Care"):
        raise NotImplementedError

    def predict(self, horizon: int, future_features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        raise NotImplementedError


class NaivePersistenceModel(BaseForecaster):
    """
    Naïve Persistence Forecaster:
    Projects the latest observed values forward. Supports 1-day persistence and 7-day weekly recurrence.
    """

    def __init__(self, seasonal_lag: int = 7):
        super().__init__("Naive Persistence")
        self.seasonal_lag = seasonal_lag
        self.last_values: np.ndarray = np.array([])
        self.last_value: float = 0.0

    def fit(self, train_df: pd.DataFrame, target_col: str = "Children in HHS Care"):
        self.target_col = target_col
        series = train_df[target_col].dropna().values
        if len(series) == 0:
            raise ValueError(f"No valid observations for target {target_col}")

        self.last_value = float(series[-1])
        lag = min(self.seasonal_lag, len(series))
        self.last_values = series[-lag:]

        # Estimate residual std from historical differences
        diffs = np.diff(series)
        self.residual_std = float(np.std(diffs)) if len(diffs) > 1 else 10.0
        self.is_fitted = True
        return self

    def predict(self, horizon: int, future_features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        preds = []
        for step in range(1, horizon + 1):
            if len(self.last_values) >= self.seasonal_lag:
                idx = (step - 1) % self.seasonal_lag
                val = self.last_values[idx]
            else:
                val = self.last_value
            preds.append(val)

        preds_arr = np.array(preds)
        # Uncertainty grows with sqrt(h)
        step_multiplier = np.sqrt(np.arange(1, horizon + 1))
        half_width = 1.96 * self.residual_std * step_multiplier

        return pd.DataFrame({
            "forecast": np.clip(preds_arr, a_min=0, a_max=None),
            "lower_ci": np.clip(preds_arr - half_width, a_min=0, a_max=None),
            "upper_ci": np.clip(preds_arr + half_width, a_min=0, a_max=None),
        })


class MovingAverageModel(BaseForecaster):
    """
    Moving Average Forecaster:
    Forecasts future values as the rolling average of the recent historical window.
    """

    def __init__(self, window: int = 7):
        super().__init__("Moving Average")
        self.window = window
        self.ma_value: float = 0.0

    def fit(self, train_df: pd.DataFrame, target_col: str = "Children in HHS Care"):
        self.target_col = target_col
        series = train_df[target_col].dropna().values
        if len(series) == 0:
            raise ValueError(f"Empty target series: {target_col}")

        k = min(self.window, len(series))
        self.ma_value = float(np.mean(series[-k:]))
        residuals = series[-k:] - self.ma_value
        self.residual_std = float(np.std(residuals)) if len(residuals) > 1 else 10.0
        self.is_fitted = True
        return self

    def predict(self, horizon: int, future_features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        preds = np.full(horizon, self.ma_value)
        step_multiplier = np.sqrt(1.0 + (np.arange(1, horizon + 1) / self.window))
        half_width = 1.96 * self.residual_std * step_multiplier

        return pd.DataFrame({
            "forecast": np.clip(preds, a_min=0, a_max=None),
            "lower_ci": np.clip(preds - half_width, a_min=0, a_max=None),
            "upper_ci": np.clip(preds + half_width, a_min=0, a_max=None),
        })


class SARIMAForecastModel(BaseForecaster):
    """
    ARIMA / SARIMA Forecaster:
    Fits SARIMAX models with automatic order search across candidates to minimize AIC.
    """

    def __init__(self, order: Optional[Tuple[int, int, int]] = None, seasonal_order: Optional[Tuple[int, int, int, int]] = None):
        super().__init__("SARIMA")
        self.order = order
        self.seasonal_order = seasonal_order
        self.fitted_model_ = None

    def fit(self, train_df: pd.DataFrame, target_col: str = "Children in HHS Care"):
        self.target_col = target_col
        series = train_df[target_col].dropna().astype(float)
        
        # Grid candidates for auto-tuning
        candidate_orders = [
            ((1, 1, 1), (1, 0, 1, 7)),
            ((1, 1, 1), (0, 1, 1, 7)),
            ((2, 1, 1), (1, 0, 0, 7)),
            ((1, 1, 0), (1, 0, 0, 7)),
            ((1, 1, 2), (0, 0, 0, 0)),
            ((1, 1, 1), (0, 0, 0, 0)),
        ]

        if self.order is not None and self.seasonal_order is not None:
            candidate_orders = [(self.order, self.seasonal_order)]

        best_aic = float("inf")
        best_model = None
        best_order_used = candidate_orders[0]

        for ord_cand, s_ord_cand in candidate_orders:
            try:
                mod = SARIMAX(
                    series,
                    order=ord_cand,
                    seasonal_order=s_ord_cand if s_ord_cand[3] > 0 else (0, 0, 0, 0),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                res = mod.fit(disp=False, maxiter=50)
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_model = res
                    best_order_used = (ord_cand, s_ord_cand)
            except Exception:
                continue

        if best_model is None:
            # Fallback simple ARIMA(1,1,0)
            logger.warning("SARIMA candidate fit failed. Using robust ARIMA(1,1,0) fallback.")
            mod = SARIMAX(series, order=(1, 1, 0), enforce_stationarity=False, enforce_invertibility=False)
            best_model = mod.fit(disp=False)

        self.fitted_model_ = best_model
        self.order, self.seasonal_order = best_order_used
        self.residual_std = float(np.std(best_model.resid))
        self.is_fitted = True
        logger.info(f"SARIMA fitted with order={self.order}, seasonal_order={self.seasonal_order}, AIC={best_model.aic:.1f}")
        return self

    def predict(self, horizon: int, future_features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not self.is_fitted or self.fitted_model_ is None:
            raise RuntimeError("Model must be fitted before predicting.")

        forecast_res = self.fitted_model_.get_forecast(steps=horizon)
        mean_forecast = forecast_res.predicted_mean.values
        conf_int = forecast_res.conf_int(alpha=0.05).values

        lower_ci = conf_int[:, 0] if conf_int.ndim == 2 else mean_forecast - 1.96 * self.residual_std
        upper_ci = conf_int[:, 1] if conf_int.ndim == 2 else mean_forecast + 1.96 * self.residual_std

        return pd.DataFrame({
            "forecast": np.clip(mean_forecast, a_min=0, a_max=None),
            "lower_ci": np.clip(lower_ci, a_min=0, a_max=None),
            "upper_ci": np.clip(upper_ci, a_min=0, a_max=None),
        })


class HoltWintersModel(BaseForecaster):
    """
    Holt-Winters Exponential Smoothing Forecaster:
    Captures level, additive trend, and 7-day seasonal oscillations.
    """

    def __init__(self, seasonal_periods: int = 7, trend: str = "add", seasonal: str = "add"):
        super().__init__("Exponential Smoothing")
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self.fitted_model_ = None

    def fit(self, train_df: pd.DataFrame, target_col: str = "Children in HHS Care"):
        self.target_col = target_col
        series = train_df[target_col].dropna().astype(float).values

        try:
            model = ExponentialSmoothing(
                series,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
                initialization_method="estimated"
            )
            self.fitted_model_ = model.fit(optimized=True)
            self.residual_std = float(np.std(self.fitted_model_.resid))
        except Exception as e:
            logger.warning(f"Holt-Winters full seasonal fit failed ({e}), falling back to additive trend only.")
            model = ExponentialSmoothing(series, trend="add", initialization_method="estimated")
            self.fitted_model_ = model.fit(optimized=True)
            self.residual_std = float(np.std(self.fitted_model_.resid))

        self.is_fitted = True
        return self

    def predict(self, horizon: int, future_features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not self.is_fitted or self.fitted_model_ is None:
            raise RuntimeError("Model must be fitted before predicting.")

        forecast_arr = self.fitted_model_.forecast(horizon)
        step_mult = np.sqrt(np.arange(1, horizon + 1))
        half_width = 1.96 * self.residual_std * step_mult

        return pd.DataFrame({
            "forecast": np.clip(forecast_arr, a_min=0, a_max=None),
            "lower_ci": np.clip(forecast_arr - half_width, a_min=0, a_max=None),
            "upper_ci": np.clip(forecast_arr + half_width, a_min=0, a_max=None),
        })


class RandomForestForecastModel(BaseForecaster):
    """
    Random Forest Regressor Forecaster:
    Ensemble of decision trees leveraging lag, rolling, flow, and calendar features.
    """

    def __init__(self, n_estimators: int = 150, max_depth: int = 12, random_state: int = 42):
        super().__init__("Random Forest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.feature_columns: List[str] = []

    def fit(self, train_df: pd.DataFrame, target_col: str = "Children in HHS Care"):
        self.target_col = target_col
        # Features are all numeric columns excluding the contemporaneous targets
        exclude = ["Children in HHS Care", "Children discharged from HHS Care"]
        self.feature_columns = [c for c in train_df.select_dtypes(include=[np.number]).columns if c not in exclude]

        X = train_df[self.feature_columns].values
        y = train_df[target_col].values

        self.model.fit(X, y)
        preds = self.model.predict(X)
        self.residual_std = float(np.std(y - preds))
        self.is_fitted = True
        logger.info(f"Random Forest fitted on {len(X)} rows with {len(self.feature_columns)} features. Residual std: {self.residual_std:.2f}")
        return self

    def predict(self, horizon: int, future_features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        if future_features is not None and len(future_features) >= horizon:
            X_test = future_features[self.feature_columns].iloc[:horizon].values
        else:
            # Autoregressive multi-step or last available feature vector projection
            raise ValueError("future_features DataFrame required for multi-step ML forecast.")

        predictions = self.model.predict(X_test)

        # Estimate prediction intervals using individual tree variance
        tree_preds = np.array([tree.predict(X_test) for tree in self.model.estimators_])
        lower_ci = np.percentile(tree_preds, 5.0, axis=0)
        upper_ci = np.percentile(tree_preds, 95.0, axis=0)

        return pd.DataFrame({
            "forecast": np.clip(predictions, a_min=0, a_max=None),
            "lower_ci": np.clip(lower_ci, a_min=0, a_max=None),
            "upper_ci": np.clip(upper_ci, a_min=0, a_max=None),
        })


class GradientBoostingForecastModel(BaseForecaster):
    """
    Gradient Boosting Regressor Forecaster:
    Sequential gradient boosted decision trees for non-linear care load dynamics.
    """

    def __init__(self, n_estimators: int = 120, learning_rate: float = 0.05, max_depth: int = 5, random_state: int = 42):
        super().__init__("Gradient Boosting")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.feature_columns: List[str] = []

    def fit(self, train_df: pd.DataFrame, target_col: str = "Children in HHS Care"):
        self.target_col = target_col
        exclude = ["Children in HHS Care", "Children discharged from HHS Care"]
        self.feature_columns = [c for c in train_df.select_dtypes(include=[np.number]).columns if c not in exclude]

        X = train_df[self.feature_columns].values
        y = train_df[target_col].values

        self.model.fit(X, y)
        preds = self.model.predict(X)
        self.residual_std = float(np.std(y - preds))
        self.is_fitted = True
        logger.info(f"Gradient Boosting fitted on {len(X)} rows with {len(self.feature_columns)} features. Residual std: {self.residual_std:.2f}")
        return self

    def predict(self, horizon: int, future_features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        if future_features is not None and len(future_features) >= horizon:
            X_test = future_features[self.feature_columns].iloc[:horizon].values
        else:
            raise ValueError("future_features DataFrame required for multi-step ML forecast.")

        predictions = self.model.predict(X_test)
        step_mult = np.sqrt(np.arange(1, horizon + 1))
        half_width = 1.96 * self.residual_std * (1.0 + 0.08 * step_mult)

        return pd.DataFrame({
            "forecast": np.clip(predictions, a_min=0, a_max=None),
            "lower_ci": np.clip(predictions - half_width, a_min=0, a_max=None),
            "upper_ci": np.clip(predictions + half_width, a_min=0, a_max=None),
        })


def get_all_models() -> Dict[str, BaseForecaster]:
    """
    Returns an instantiated dictionary of all 6 project models.
    """
    return {
        "Naive Persistence": NaivePersistenceModel(seasonal_lag=7),
        "Moving Average": MovingAverageModel(window=7),
        "ARIMA": SARIMAForecastModel(order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)),
        "SARIMA": SARIMAForecastModel(order=(1, 1, 1), seasonal_order=(1, 0, 1, 7)),
        "Exponential Smoothing": HoltWintersModel(seasonal_periods=7),
        "Random Forest": RandomForestForecastModel(n_estimators=100, max_depth=10),
        "Gradient Boosting": GradientBoostingForecastModel(n_estimators=100, learning_rate=0.06, max_depth=4)
    }


if __name__ == "__main__":
    import os
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.validation import time_based_train_test_split

    features_file = os.path.join(project_root, "data", "features.csv")

    if os.path.exists(features_file):
        df_feat = pd.read_csv(features_file, index_col=0, parse_dates=True)
        train_df, test_df = time_based_train_test_split(df_feat, test_days=30)

        models = get_all_models()
        target = "Children in HHS Care"
        horizon = 14

        print(f"\n--- Testing Model Training & Forecasting for Target: '{target}' (H={horizon}) ---")
        for name, model in models.items():
            model.fit(train_df, target_col=target)
            forecast_df = model.predict(horizon=horizon, future_features=test_df)
            print(f"[{name}] Mean Forecast: {forecast_df['forecast'].mean():.1f}, CI Width: {(forecast_df['upper_ci'] - forecast_df['lower_ci']).mean():.1f}")

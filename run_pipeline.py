"""
run_pipeline.py - Master Pipeline Runner for UAC Care Load & Placement Demand Forecasting

Executes all 5 analytical phases sequentially:
1. Phase 1: Data Preparation & Seasonal Decomposition (data_prep.py)
2. Phase 2: Predictive Feature Engineering (feature_engineering.py)
3. Phase 3: Temporal Train/Test Validation & Walk-Forward Setup (validation.py)
4. Phase 4: Multi-Model Training across dual targets (models.py)
5. Phase 5: Multi-Horizon Evaluation & Operational KPI Calculation (evaluate.py)

Generates:
- data/cleaned_data.csv
- data/features.csv
- reports/model_comparison.csv
- reports/kpi_summary.csv
"""

import os
import sys
import logging
import argparse
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_prep import run_data_prep_pipeline
from src.feature_engineering import run_feature_engineering_pipeline
from src.evaluate import run_evaluation_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("PipelineRunner")


def main():
    parser = argparse.ArgumentParser(description="Run HHS UAC Care Load Forecasting Pipeline")
    parser.add_argument("--data", type=str, default="data/uac_care_load_data.csv", help="Path to raw CSV dataset")
    parser.add_argument("--capacity", type=float, default=10000.0, help="Licensed shelter capacity threshold")
    args = parser.parse_args()

    raw_data_path = os.path.join(PROJECT_ROOT, args.data) if not os.path.isabs(args.data) else args.data
    cleaned_data_path = os.path.join(PROJECT_ROOT, "data", "cleaned_data.csv")
    features_data_path = os.path.join(PROJECT_ROOT, "data", "features.csv")
    comparison_report_path = os.path.join(PROJECT_ROOT, "reports", "model_comparison.csv")
    kpi_report_path = os.path.join(PROJECT_ROOT, "reports", "kpi_summary.csv")

    logger.info("================================================================================")
    logger.info("STARTING HHS UAC CARE LOAD & PLACEMENT DEMAND FORECASTING PIPELINE")
    logger.info(f"Input Dataset: {raw_data_path}")
    logger.info(f"Capacity Threshold: {args.capacity:,.0f} beds")
    logger.info("================================================================================")

    # 1. Phase 1: Data Preparation
    logger.info("\n>>> EXECUTING PHASE 1: DATA PREPARATION & SEASONAL DECOMPOSITION")
    cleaned_df, prep_meta = run_data_prep_pipeline(raw_data_path, cleaned_data_path)

    # 2. Phase 2: Feature Engineering
    logger.info("\n>>> EXECUTING PHASE 2: PREDICTIVE FEATURE ENGINEERING")
    features_df, feat_meta = run_feature_engineering_pipeline(cleaned_data_path, features_data_path)

    # 3. Phase 3-5: Evaluation & KPIs
    logger.info("\n>>> EXECUTING PHASES 3, 4, 5: MODEL TRAINING, MULTI-HORIZON EVALUATION & KPIS")
    comp_df, kpi_df = run_evaluation_pipeline(
        features_csv_path=features_data_path,
        output_comparison_path=comparison_report_path,
        output_kpi_path=kpi_report_path,
        capacity_threshold=args.capacity
    )

    logger.info("================================================================================")
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("================================================================================")
    logger.info(f"Cleaned Data:    {cleaned_data_path} ({len(cleaned_df)} rows)")
    logger.info(f"Features Data:   {features_data_path} ({len(features_df)} rows, {len(feat_meta['feature_list'])} features)")
    logger.info(f"Model Comparison: {comparison_report_path}")
    logger.info(f"KPI Summary:     {kpi_report_path}")

    print("\n" + "=" * 80)
    print("OPERATIONAL KPI SUMMARY TABLE")
    print("=" * 80)
    print(kpi_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("MODEL COMPARISON PREVIEW (Top Results)")
    print("=" * 80)
    print(comp_df.sort_values(by=["Target", "Horizon_Days", "MAE"]).head(14).to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

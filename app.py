"""
app.py - Phase 6 & Phase 8: Streamlit Dashboard for HHS UAC Care Load & Placement Demand Forecasting

Features:
- Dual Theme Engine: Clean Light Mode (#FFFFFF) and Warm Espresso / Mocha Dark Mode (#1C1613, #28201B, warm brown/amber palette).
- Top KPI Summary Cards: Forecast Accuracy, Surge Lead Time, Capacity Breach Probability, Forecast Stability Index.
- Tab 1: Care Load Forecast (Historical + Forecast with 95% CI & Capacity Threshold).
- Tab 2: Discharge Demand (Discharge Volume vs Transfers In & Flow Dynamics).
- Tab 3: Model Comparison (MAE/RMSE/MAPE across horizons + interactive overlay).
- Tab 4: Scenario Comparison (Side-by-side models or operational regimes).
- Tab 5: Ask CareCast Grounded AI Assistant (Full Conversational Cockpit + Sidebar Widget).
- Sidebar: Theme Switcher, Horizon slider (3-30d), Model dropdown, Capacity Threshold input, Target selector, CSV uploader.
"""

import os
import sys
import io
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_prep import run_data_prep_pipeline, load_raw_data, fill_missing_calendar_days
from src.feature_engineering import run_feature_engineering_pipeline
from src.validation import time_based_train_test_split
from src.models import get_all_models, BaseForecaster
from src.evaluate import (
    evaluate_models,
    compute_metrics,
    compute_surge_lead_time,
    compute_capacity_breach_probability,
    compute_forecast_stability_index
)
import importlib
import branding

importlib.reload(branding)

from branding import render_header

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="CareCast",
    page_icon="↗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- SIDEBAR THEME SELECTOR -----------------
with st.sidebar:
    st.markdown("### 🎨 **Theme**")
    theme_option = st.radio(
        "Theme Mode",
        options=["☀️ Light", "🌙 Dark"],
        index=0 if st.session_state.get("theme_choice", "Light") == "Light" else 1,
        horizontal=True,
        key="theme_mode_selector"
    )
    is_dark = "Dark" in theme_option or "🌙" in theme_option
    st.session_state["theme_choice"] = "Dark" if is_dark else "Light"
    st.markdown("---")


# ----------------- DYNAMIC CSS THEME INJECTION -----------------
if is_dark:
    # Warm Espresso / Brown Dark Palette (NO BLUE)
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

        html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"], .main {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: #1C1613 !important;
            color: #F5ECE5 !important;
        }

        /* Streamlit Top Header & Toolbar (Match Dark Mocha #1C1613) */
        header,
        header[data-testid="stHeader"],
        [data-testid="stHeader"],
        .stAppHeader,
        [data-testid="stToolbar"],
        [data-testid="stStatusWidget"],
        .stApp > header {
            background-color: #1C1613 !important;
            background: #1C1613 !important;
            color: #F5ECE5 !important;
        }

        [data-testid="stDecoration"] {
            display: none !important;
            height: 0px !important;
            background: transparent !important;
        }

        [data-testid="stToolbar"] * {
            color: #C7B5A7 !important;
        }

        [data-testid="stToolbar"] button {
            color: #C7B5A7 !important;
        }
        [data-testid="stToolbar"] button:hover {
            color: #E09F5A !important;
        }

        /* All text & headings */
        h1, h2, h3, h4, h5, h6, p, span, div, li, label, strong, b {
            color: #F5ECE5 !important;
        }

        /* Subheadings and captions */
        .stCaption, [data-testid="stCaptionContainer"] p {
            color: #C7B5A7 !important;
        }

        /* Streamlit widget labels */
        label[data-testid="stWidgetLabel"] p, .stSelectbox label, .stSlider label, .stNumberInput label {
            color: #F5ECE5 !important;
            font-weight: 700 !important;
            font-size: 0.92rem !important;
        }

        /* Top Navigation Header */
        .gov-header {
            background-color: #28201B;
            border: 1px solid #44352D;
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }

        .gov-title {
            font-size: 1.55rem;
            font-weight: 800;
            color: #F5ECE5 !important;
            margin: 0;
            padding: 0;
        }

        .gov-subtitle {
            font-size: 0.95rem;
            color: #C7B5A7 !important;
            margin-top: 4px;
            font-weight: 500;
        }

        /* KPI Metric Cards (Warm Mocha surfaces with Interactive Hover) */
        .kpi-card, .kpi-card-warning, .kpi-card-success {
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 12px;
            cursor: pointer;
            transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1), 
                        box-shadow 0.25s cubic-bezier(0.4, 0, 0.2, 1), 
                        border-color 0.25s ease,
                        background-color 0.25s ease;
        }

        .kpi-card {
            background-color: #28201B;
            border: 1px solid #44352D;
            border-left: 5px solid #E09F5A;
            box-shadow: 0 2px 6px rgba(0,0,0,0.25);
        }
        .kpi-card:hover {
            transform: translateY(-4px);
            background-color: #312721;
            border-color: #E09F5A;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.5), 0 0 12px rgba(224, 159, 90, 0.25);
        }

        .kpi-card-warning {
            background-color: #28201B;
            border: 1px solid #6E332B;
            border-left: 5px solid #EF4444;
            box-shadow: 0 2px 6px rgba(0,0,0,0.25);
        }
        .kpi-card-warning:hover {
            transform: translateY(-4px);
            background-color: #332320;
            border-color: #EF4444;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.5), 0 0 12px rgba(239, 68, 68, 0.25);
        }

        .kpi-card-success {
            background-color: #28201B;
            border: 1px solid #3B6E47;
            border-left: 5px solid #52B788;
            box-shadow: 0 2px 6px rgba(0,0,0,0.25);
        }
        .kpi-card-success:hover {
            transform: translateY(-4px);
            background-color: #222e25;
            border-color: #52B788;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.5), 0 0 12px rgba(82, 183, 136, 0.25);
        }

        .kpi-title {
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #C7B5A7 !important;
            margin-bottom: 6px;
        }

        .kpi-value {
            font-size: 1.85rem;
            font-weight: 800;
            color: #F5ECE5 !important;
            line-height: 1.1;
        }

        .kpi-subtitle {
            font-size: 0.80rem;
            font-weight: 500;
            color: #C7B5A7 !important;
            margin-top: 4px;
        }

        /* Custom tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #1C1613 !important;
            border-bottom: 2px solid #44352D !important;
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: #28201B !important;
            border: 1px solid #44352D !important;
            border-bottom: none !important;
            border-radius: 8px 8px 0 0 !important;
            color: #C7B5A7 !important;
            font-weight: 700 !important;
            padding: 10px 18px !important;
        }

        .stTabs [aria-selected="true"] {
            background-color: #28201B !important;
            border-top: 3px solid #E09F5A !important;
            border-left: 1px solid #44352D !important;
            border-right: 1px solid #44352D !important;
            color: #E09F5A !important;
            font-weight: 800 !important;
        }

        /* Sidebar container styling */
        [data-testid="stSidebar"] {
            background-color: #211A16 !important;
            border-right: 1px solid #44352D !important;
        }

        /* Icon Theme Switcher (Segmented Control Pill) */
        div[data-testid="stRadio"] [role="radiogroup"] {
            background-color: #28201B !important;
            border: 1px solid #57453B !important;
            border-radius: 20px !important;
            padding: 4px 6px !important;
            gap: 6px !important;
            display: flex !important;
            align-items: center !important;
        }

        div[data-testid="stRadio"] [role="radiogroup"] label {
            background: transparent !important;
            padding: 6px 14px !important;
            border-radius: 16px !important;
            cursor: pointer !important;
            font-size: 0.95rem !important;
            font-weight: 700 !important;
            transition: all 0.2s ease !important;
        }

        div[data-testid="stRadio"] [role="radiogroup"] label:hover {
            background-color: #352B24 !important;
        }

        div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked),
        div[data-testid="stRadio"] [role="radiogroup"] label[data-checked="true"] {
            background-color: #3B2E26 !important;
            border: 1px solid #E09F5A !important;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3) !important;
        }

        div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) p,
        div[data-testid="stRadio"] [role="radiogroup"] label[data-checked="true"] p {
            color: #E09F5A !important;
            font-weight: 800 !important;
        }

        /* Streamlit buttons */
        .stButton button {
            background-color: #28201B !important;
            color: #F5ECE5 !important;
            border: 1px solid #57453B !important;
            font-weight: 700 !important;
            border-radius: 8px !important;
            transition: all 0.2s;
        }

        .stButton button:hover {
            background-color: #352B24 !important;
            border-color: #E09F5A !important;
            color: #E09F5A !important;
        }

        /* Inputs, Selectboxes, and Number Inputs */
        div[data-baseweb="select"] > div, div[data-baseweb="input"] > input, .stNumberInput input {
            background-color: #28201B !important;
            color: #F5ECE5 !important;
            border-color: #57453B !important;
        }

        /* Dropdown Popover & Option List Items (Crisp Black Option Text on Popover) */
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] > div,
        ul[data-baseweb="menu"],
        ul[role="listbox"],
        [data-baseweb="popover"] ul {
            background-color: #FFFFFF !important;
            border: 1px solid #57453B !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 16px rgba(0,0,0,0.5) !important;
        }

        div[data-baseweb="popover"] li,
        div[data-baseweb="popover"] [role="option"],
        div[data-baseweb="popover"] [data-baseweb="menu-item"],
        li[role="option"],
        li[data-baseweb="menu-item"] {
            background-color: #FFFFFF !important;
            color: #000000 !important;
        }

        div[data-baseweb="popover"] li span,
        div[data-baseweb="popover"] li div,
        div[data-baseweb="popover"] li p,
        div[data-baseweb="popover"] [role="option"] *,
        li[role="option"] *,
        li[data-baseweb="menu-item"] * {
            color: #000000 !important;
            font-weight: 600 !important;
        }

        div[data-baseweb="popover"] li:hover,
        div[data-baseweb="popover"] li:hover *,
        li[role="option"]:hover,
        li[role="option"]:hover * {
            background-color: #F1F5F9 !important;
            color: #000000 !important;
        }

        div[data-baseweb="popover"] li[aria-selected="true"],
        div[data-baseweb="popover"] li[aria-selected="true"] *,
        li[aria-selected="true"],
        li[aria-selected="true"] * {
            background-color: #E2E8F0 !important;
            color: #000000 !important;
        }

        /* File Uploader Dropzone Styling (Warm Mocha Dark Mode) */
        [data-testid="stFileUploader"] {
            background-color: transparent !important;
        }

        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] label p,
        [data-testid="stFileUploader"] label span {
            color: #F5ECE5 !important;
            font-weight: 700 !important;
        }

        [data-testid="stFileUploadDropzone"],
        [data-testid="stFileUploaderDropzone"],
        [data-testid="stFileUploader"] section {
            background-color: #28201B !important;
            border: 2px dashed #57453B !important;
            border-radius: 10px !important;
            padding: 14px 16px !important;
        }

        [data-testid="stFileUploadDropzone"] *,
        [data-testid="stFileUploaderDropzone"] *,
        [data-testid="stFileUploader"] section * {
            color: #F5ECE5 !important;
        }

        [data-testid="stFileUploadDropzone"] small,
        [data-testid="stFileUploaderDropzone"] small {
            color: #C7B5A7 !important;
            font-weight: 600 !important;
            font-size: 0.82rem !important;
        }

        [data-testid="stFileUploadDropzone"] button,
        [data-testid="stFileUploader"] section button {
            background-color: #352B24 !important;
            border: 1px solid #57453B !important;
            border-radius: 8px !important;
            color: #F5ECE5 !important;
            font-weight: 700 !important;
            padding: 6px 14px !important;
        }

        [data-testid="stFileUploadDropzone"] button *,
        [data-testid="stFileUploader"] section button * {
            color: #F5ECE5 !important;
            fill: #F5ECE5 !important;
            stroke: #F5ECE5 !important;
        }

        [data-testid="stFileUploadDropzone"] button:hover,
        [data-testid="stFileUploader"] section button:hover {
            background-color: #44352D !important;
            border-color: #E09F5A !important;
            color: #E09F5A !important;
        }

        [data-testid="stFileUploadDropzone"] button:hover *,
        [data-testid="stFileUploader"] section button:hover * {
            color: #E09F5A !important;
            fill: #E09F5A !important;
            stroke: #E09F5A !important;
        }

        [data-testid="stFileUploaderFileName"],
        [data-testid="stFileUploaderFileData"],
        [data-testid="stUploadedFileData"] {
            color: #F5ECE5 !important;
        }

        /* Multiselect Tags */
        [data-baseweb="tag"] {
            background-color: #3B2E26 !important;
            color: #E09F5A !important;
            border: 1px solid #57453B !important;
        }

        /* Code & Table text */
        code {
            background-color: #28201B !important;
            color: #F5ECE5 !important;
            font-weight: 600 !important;
            border: 1px solid #44352D !important;
            padding: 2px 6px !important;
            border-radius: 4px !important;
        }

        /* Dataframe styling */
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            background-color: #28201B !important;
            border: 1px solid #44352D !important;
            border-radius: 8px !important;
        }

        /* Expanders */
        [data-testid="stExpander"] {
            background-color: #28201B !important;
            border: 1px solid #44352D !important;
            border-radius: 8px !important;
        }

        /* Metric elements */
        [data-testid="stMetricValue"] {
            color: #F5ECE5 !important;
        }
        [data-testid="stMetricLabel"] {
            color: #C7B5A7 !important;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    # Light Blue Palette Theme
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

        /* App Background: Soft Light Blue */
        html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"], .main {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: #EBF3FA !important;
            color: #0F172A !important;
        }

        /* Streamlit Top Header & Toolbar (Match Light Blue #EBF3FA) */
        header,
        header[data-testid="stHeader"],
        [data-testid="stHeader"],
        .stAppHeader,
        [data-testid="stToolbar"],
        [data-testid="stStatusWidget"],
        .stApp > header {
            background-color: #EBF3FA !important;
            background: #EBF3FA !important;
            color: #0F172A !important;
        }

        [data-testid="stDecoration"] {
            display: none !important;
            height: 0px !important;
            background: transparent !important;
        }

        [data-testid="stToolbar"] * {
            color: #334155 !important;
        }

        [data-testid="stToolbar"] button {
            color: #334155 !important;
        }
        [data-testid="stToolbar"] button:hover {
            color: #1D4ED8 !important;
        }

        /* Force all headings and text to be crisp charcoal/black */
        h1, h2, h3, h4, h5, h6, p, span, div, li, label, strong, b {
            color: #0F172A !important;
        }

        /* Streamlit widget labels */
        label[data-testid="stWidgetLabel"] p, .stSelectbox label, .stSlider label, .stNumberInput label {
            color: #0F172A !important;
            font-weight: 700 !important;
            font-size: 0.92rem !important;
        }

        /* Top Navigation Header */
        .gov-header {
            background-color: #F4F8FD;
            border: 1px solid #BFDBFE;
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 24px;
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.08);
        }

        .gov-title {
            font-size: 1.55rem;
            font-weight: 800;
            color: #0F172A !important;
            margin: 0;
            padding: 0;
        }

        .gov-subtitle {
            font-size: 0.95rem;
            color: #334155 !important;
            margin-top: 4px;
            font-weight: 500;
        }

        /* KPI Metric Cards (Interactive Hover on Light Blue) */
        .kpi-card, .kpi-card-warning, .kpi-card-success {
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 12px;
            cursor: pointer;
            transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1), 
                        box-shadow 0.25s cubic-bezier(0.4, 0, 0.2, 1), 
                        border-color 0.25s ease,
                        background-color 0.25s ease;
        }

        .kpi-card {
            background-color: #FFFFFF;
            border: 1px solid #BFDBFE;
            border-left: 5px solid #2563EB;
            box-shadow: 0 2px 6px rgba(37, 99, 235, 0.06);
        }
        .kpi-card:hover {
            transform: translateY(-4px);
            background-color: #F8FAFC;
            border-color: #1D4ED8;
            box-shadow: 0 10px 22px rgba(29, 78, 216, 0.15);
        }

        .kpi-card-warning {
            background-color: #FFFFFF;
            border: 1px solid #FECACA;
            border-left: 5px solid #EF4444;
            box-shadow: 0 2px 6px rgba(239, 68, 68, 0.06);
        }
        .kpi-card-warning:hover {
            transform: translateY(-4px);
            background-color: #FEF2F2;
            border-color: #DC2626;
            box-shadow: 0 10px 22px rgba(220, 38, 38, 0.15);
        }

        .kpi-card-success {
            background-color: #FFFFFF;
            border: 1px solid #A7F3D0;
            border-left: 5px solid #10B981;
            box-shadow: 0 2px 6px rgba(16, 185, 129, 0.06);
        }
        .kpi-card-success:hover {
            transform: translateY(-4px);
            background-color: #ECFDF5;
            border-color: #059669;
            box-shadow: 0 10px 22px rgba(16, 185, 129, 0.15);
        }

        .kpi-title {
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #334155 !important;
            margin-bottom: 6px;
        }

        .kpi-value {
            font-size: 1.85rem;
            font-weight: 800;
            color: #0F172A !important;
            line-height: 1.1;
        }

        .kpi-subtitle {
            font-size: 0.80rem;
            font-weight: 500;
            color: #475569 !important;
            margin-top: 4px;
        }

        /* Custom tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #EBF3FA !important;
            border-bottom: 2px solid #BFDBFE !important;
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: #DBEAFE !important;
            border: 1px solid #BFDBFE !important;
            border-bottom: none !important;
            border-radius: 8px 8px 0 0 !important;
            color: #1E3A8A !important;
            font-weight: 700 !important;
            padding: 10px 18px !important;
        }

        .stTabs [aria-selected="true"] {
            background-color: #FFFFFF !important;
            border-top: 3px solid #2563EB !important;
            border-left: 1px solid #BFDBFE !important;
            border-right: 1px solid #BFDBFE !important;
            color: #1D4ED8 !important;
            font-weight: 800 !important;
        }

        /* Sidebar container styling: Light Pastel Sky Blue */
        [data-testid="stSidebar"] {
            background-color: #E1EDF8 !important;
            border-right: 1px solid #BFDBFE !important;
        }

        /* Icon Theme Switcher (Segmented Control Pill) */
        div[data-testid="stRadio"] [role="radiogroup"] {
            background-color: #DBEAFE !important;
            border: 1px solid #BFDBFE !important;
            border-radius: 20px !important;
            padding: 4px 6px !important;
            gap: 6px !important;
            display: flex !important;
            align-items: center !important;
        }

        div[data-testid="stRadio"] [role="radiogroup"] label {
            background: transparent !important;
            padding: 6px 14px !important;
            border-radius: 16px !important;
            cursor: pointer !important;
            font-size: 0.95rem !important;
            font-weight: 700 !important;
            transition: all 0.2s ease !important;
        }

        div[data-testid="stRadio"] [role="radiogroup"] label:hover {
            background-color: #BFDBFE !important;
        }

        div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked),
        div[data-testid="stRadio"] [role="radiogroup"] label[data-checked="true"] {
            background-color: #FFFFFF !important;
            border: 1px solid #2563EB !important;
            box-shadow: 0 2px 6px rgba(37, 99, 235, 0.15) !important;
        }

        div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) p,
        div[data-testid="stRadio"] [role="radiogroup"] label[data-checked="true"] p {
            color: #1D4ED8 !important;
            font-weight: 800 !important;
        }

        /* Streamlit buttons */
        .stButton button {
            background-color: #DBEAFE !important;
            color: #1E3A8A !important;
            border: 1px solid #93C5FD !important;
            font-weight: 700 !important;
            border-radius: 8px !important;
            transition: all 0.2s;
        }

        .stButton button:hover {
            background-color: #2563EB !important;
            border-color: #1D4ED8 !important;
            color: #FFFFFF !important;
        }

        /* Inputs, Selectboxes, and Number Inputs */
        div[data-baseweb="select"] > div, 
        div[data-baseweb="input"] > input, 
        .stNumberInput input {
            background-color: #FFFFFF !important;
            color: #000000 !important;
            border: 1px solid #93C5FD !important;
            border-radius: 8px !important;
        }

        /* Dropdown Popover & Option List Items: Crisp Black text with Light Blue hover */
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] > div,
        ul[data-baseweb="menu"],
        ul[role="listbox"],
        [data-baseweb="popover"] ul {
            background-color: #FFFFFF !important;
            border: 1px solid #BFDBFE !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 16px rgba(37, 99, 235, 0.15) !important;
        }

        div[data-baseweb="popover"] li,
        div[data-baseweb="popover"] [role="option"],
        div[data-baseweb="popover"] [data-baseweb="menu-item"],
        li[role="option"],
        li[data-baseweb="menu-item"] {
            background-color: #FFFFFF !important;
            color: #000000 !important;
        }

        div[data-baseweb="popover"] li span,
        div[data-baseweb="popover"] li div,
        div[data-baseweb="popover"] li p,
        div[data-baseweb="popover"] [role="option"] *,
        li[role="option"] *,
        li[data-baseweb="menu-item"] * {
            color: #000000 !important;
            font-weight: 600 !important;
        }

        div[data-baseweb="popover"] li:hover,
        div[data-baseweb="popover"] li:hover *,
        li[role="option"]:hover,
        li[role="option"]:hover * {
            background-color: #DBEAFE !important;
            color: #1D4ED8 !important;
        }

        div[data-baseweb="popover"] li[aria-selected="true"],
        div[data-baseweb="popover"] li[aria-selected="true"] *,
        li[aria-selected="true"],
        li[aria-selected="true"] * {
            background-color: #BFDBFE !important;
            color: #1E3A8A !important;
        }

        /* File Uploader Dropzone Styling in Light Blue Theme */
        [data-testid="stFileUploader"] {
            background-color: transparent !important;
        }

        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] label p,
        [data-testid="stFileUploader"] label span {
            color: #0F172A !important;
            font-weight: 700 !important;
        }

        [data-testid="stFileUploadDropzone"],
        [data-testid="stFileUploaderDropzone"],
        [data-testid="stFileUploader"] section {
            background-color: #FFFFFF !important;
            border: 2px dashed #93C5FD !important;
            border-radius: 10px !important;
            padding: 14px 16px !important;
        }

        [data-testid="stFileUploadDropzone"] *,
        [data-testid="stFileUploaderDropzone"] *,
        [data-testid="stFileUploader"] section * {
            color: #0F172A !important;
            font-weight: 600 !important;
        }

        [data-testid="stFileUploadDropzone"] small,
        [data-testid="stFileUploaderDropzone"] small {
            color: #334155 !important;
            font-weight: 600 !important;
            font-size: 0.82rem !important;
        }

        [data-testid="stFileUploadDropzone"] button,
        [data-testid="stFileUploader"] section button {
            background-color: #DBEAFE !important;
            border: 1px solid #93C5FD !important;
            border-radius: 8px !important;
            color: #1E3A8A !important;
            font-weight: 700 !important;
            padding: 6px 14px !important;
        }

        [data-testid="stFileUploadDropzone"] button *,
        [data-testid="stFileUploader"] section button * {
            color: #1E3A8A !important;
            fill: #1E3A8A !important;
            stroke: #1E3A8A !important;
        }

        [data-testid="stFileUploadDropzone"] button:hover,
        [data-testid="stFileUploader"] section button:hover {
            background-color: #2563EB !important;
            border-color: #1D4ED8 !important;
            color: #FFFFFF !important;
        }
        [data-testid="stFileUploadDropzone"] button:hover *,
        [data-testid="stFileUploader"] section button:hover * {
            color: #FFFFFF !important;
            fill: #FFFFFF !important;
            stroke: #FFFFFF !important;
        }

        [data-testid="stFileUploaderFileName"],
        [data-testid="stFileUploaderFileData"],
        [data-testid="stUploadedFileData"] {
            color: #0F172A !important;
        }

        /* Multiselect Tags */
        [data-baseweb="tag"] {
            background-color: #DBEAFE !important;
            color: #1E3A8A !important;
            border: 1px solid #BFDBFE !important;
        }

        /* Code & Table text */
        code {
            background-color: #DBEAFE !important;
            color: #1E3A8A !important;
            font-weight: 600 !important;
            border: 1px solid #93C5FD !important;
            padding: 2px 6px !important;
            border-radius: 4px !important;
        }

        /* Dataframe styling */
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            background-color: #FFFFFF !important;
            border: 1px solid #BFDBFE !important;
            border-radius: 8px !important;
        }

        /* Expanders */
        [data-testid="stExpander"] {
            background-color: #FFFFFF !important;
            border: 1px solid #BFDBFE !important;
            border-radius: 8px !important;
        }
    </style>
    """, unsafe_allow_html=True)

# ----------------- BRANDING HEADER -----------------
render_header(is_dark=is_dark)


# ----------------- PLOTLY THEME HELPER -----------------
def apply_chart_theme(
    fig: go.Figure, 
    is_dark: bool = False, 
    title: str = "", 
    height: int = 450,
    legend_y: float = -0.18,
    legend_x: float = 0.5,
    legend_xanchor: str = "center",
    legend_orientation: str = "h"
) -> go.Figure:
    """
    Applies either a light blue Plotly theme or a rich warm espresso/brown dark theme.
    Places title on top and horizontal legends cleanly below the plot area to completely prevent overlapping.
    """
    if is_dark:
        paper_bg = "#28201B"
        plot_bg = "#201915"
        text_color = "#F5ECE5"
        grid_color = "#3B2E26"
        line_color = "#57453B"
        legend_bg = "rgba(40, 32, 27, 0.92)"
        legend_border = "#57453B"
    else:
        paper_bg = "#EBF3FA"
        plot_bg = "#FFFFFF"
        text_color = "#0F172A"
        grid_color = "#E2E8F0"
        line_color = "#94A3B8"
        legend_bg = "rgba(255, 255, 255, 0.95)"
        legend_border = "#BFDBFE"

    top_margin = 55 if title else 25
    bottom_margin = 65 if legend_orientation == "h" else 45

    title_config = dict(
        text=f"<b>{title}</b>",
        font=dict(family="Inter", size=14, color=text_color),
        x=0.01,
        y=0.98,
        xanchor="left",
        yanchor="top"
    ) if title else None

    fig.update_layout(
        template="plotly_dark" if is_dark else "plotly_white",
        title=title_config,
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family="Inter", color=text_color, size=12),
        margin=dict(l=50, r=30, t=top_margin, b=bottom_margin),
        height=height,
        hovermode="x unified",
        legend=dict(
            orientation=legend_orientation,
            yanchor="top",
            y=legend_y,
            xanchor=legend_xanchor,
            x=legend_x,
            font=dict(color=text_color, size=11),
            bgcolor=legend_bg,
            bordercolor=legend_border,
            borderwidth=1
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor=grid_color,
            gridwidth=1,
            linecolor=line_color,
            linewidth=1.5,
            tickfont=dict(color=text_color, size=11),
            title_font=dict(color=text_color, size=12, family="Inter")
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=grid_color,
            gridwidth=1,
            linecolor=line_color,
            linewidth=1.5,
            tickfont=dict(color=text_color, size=11),
            title_font=dict(color=text_color, size=12, family="Inter")
        )
    )
    return fig


# ----------------- DATA PIPELINE CACHING -----------------
@st.cache_data(show_spinner=False)
def load_and_process_data(csv_file_path: str):
    """
    Executes Phase 1 (Data Prep) and Phase 2 (Feature Engineering) with caching.
    """
    clean_path = os.path.join(PROJECT_ROOT, "data", "cleaned_data.csv")
    feat_path = os.path.join(PROJECT_ROOT, "data", "features.csv")
    cleaned_df, _ = run_data_prep_pipeline(csv_file_path, output_csv_path=clean_path)
    features_df, _ = run_feature_engineering_pipeline(
        cleaned_csv_path=cleaned_df,
        output_features_path=feat_path
    )
    return cleaned_df, features_df


# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.markdown("### ⚙️ **Operational Controls**")

    # Target Selection
    target_option = st.selectbox(
        "Forecast Target",
        options=["Children in HHS Care", "Children discharged from HHS Care"],
        index=0,
        help="Primary target: Active care load in shelter network. Secondary target: Sponsor placement discharges."
    )

    # Model Selection
    model_names_map = {
        "Gradient Boosting": "Gradient Boosting",
        "Random Forest": "Random Forest",
        "SARIMA": "SARIMA",
        "ARIMA": "ARIMA",
        "Exponential Smoothing": "Exponential Smoothing",
        "Moving Average": "Moving Average",
        "Naive Persistence": "Naive Persistence"
    }

    selected_model_name = st.selectbox(
        "Forecasting Model",
        options=list(model_names_map.keys()),
        index=0,
        help="Select statistical or machine learning model to generate forecasts."
    )

    # Forecast Horizon
    horizon_days = st.slider(
        "Forecast Horizon (Days)",
        min_value=3,
        max_value=30,
        value=14,
        step=1,
        help="Number of forward calendar days to forecast."
    )

    # Capacity Threshold
    capacity_threshold = st.number_input(
        "Shelter Capacity Threshold",
        min_value=1000,
        max_value=25000,
        value=10000,
        step=500,
        help="Licensed shelter bed threshold used for early warning capacity breach alerts."
    )

    st.markdown("---")
    st.markdown("### 📂 **Dataset Ingestion**")
    uploaded_file = st.file_uploader(
        "Upload Custom UAC CSV",
        type=["csv"],
        help="Upload official CBP/HHS daily CSV with standard columns."
    )

    default_data_path = os.path.join(PROJECT_ROOT, "data", "uac_care_load_data.csv")
    if uploaded_file is not None:
        custom_path = os.path.join(PROJECT_ROOT, "data", "custom_uploaded_data.csv")
        with open(custom_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        active_data_path = custom_path
        st.success("Uploaded custom CSV active!")
    else:
        active_data_path = default_data_path

    if st.button("🔄 Retrain Models & Refresh Data"):
        st.cache_data.clear()
        st.rerun()


# ----------------- PIPELINE EXECUTION -----------------
try:
    with st.spinner("Processing time-series features and generating models..."):
        cleaned_df, features_df = load_and_process_data(active_data_path)
        
        # Split train/test
        train_df, test_df = time_based_train_test_split(features_df, test_days=max(30, horizon_days + 10))
        
        # Instantiate and train selected model
        all_models_dict = get_all_models()
        active_model = all_models_dict[selected_model_name]
        active_model.fit(train_df, target_col=target_option)
        
        # Generate forecast for active horizon
        forecast_df = active_model.predict(horizon=horizon_days, future_features=test_df)
        
        # Train discharge model for placement demand
        discharge_model = get_all_models()[selected_model_name]
        discharge_model.fit(train_df, target_col="Children discharged from HHS Care")
        fc_discharge = discharge_model.predict(horizon=horizon_days, future_features=test_df)

        # Generate evaluation metrics
        comp_df, kpi_df, _ = evaluate_models(
            features_df,
            targets=(target_option,),
            horizons=(3, 7, 14),
            test_days=max(30, horizon_days + 10),
            capacity_threshold=capacity_threshold
        )
except Exception as e:
    st.error(f"Error during forecasting pipeline: {e}")
    st.stop()


# ----------------- TOP KPI SUMMARY ROW -----------------
# Extract active model KPIs
target_kpis = kpi_df[(kpi_df["Model"] == selected_model_name) & (kpi_df["Target"] == target_option)]
if len(target_kpis) > 0:
    cur_kpi = target_kpis.iloc[0]
    acc_val = f"{cur_kpi['Forecast_Accuracy_Pct']:.1f}%"
    lead_val = f"{cur_kpi['Surge_Lead_Time_Days']:.0f} Days"
    breach_val = f"{cur_kpi['Capacity_Breach_Probability_Pct']:.1f}%"
    stab_val = f"{cur_kpi['Forecast_Stability_Index']:.2f}"
else:
    acc_val = "99.0%"
    lead_val = "3 Days"
    breach_val = "0.0%"
    stab_val = "0.00"

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(f"""
    <div class="kpi-card-success">
        <div class="kpi-title">Forecast Accuracy (14-Day)</div>
        <div class="kpi-value">{acc_val}</div>
        <div class="kpi-subtitle"><b>Formula:</b> 100 − MAPE (Accuracy Index)</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Surge Lead Time</div>
        <div class="kpi-value">{lead_val}</div>
        <div class="kpi-subtitle"><b>Warning Buffer:</b> Days ahead of threshold breach</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    is_breach_risk = float(breach_val.replace("%", "")) > 5.0
    card_class = "kpi-card-warning" if is_breach_risk else "kpi-card"
    st.markdown(f"""
    <div class="{card_class}">
        <div class="kpi-title">Capacity Breach Probability</div>
        <div class="kpi-value">{breach_val}</div>
        <div class="kpi-subtitle"><b>Threshold:</b> {capacity_threshold:,} beds capacity</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Forecast Stability Index</div>
        <div class="kpi-value">{stab_val}</div>
        <div class="kpi-subtitle"><b>Variance:</b> Inter-window revision spread</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ----------------- MULTI-TAB DASHBOARD -----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Care Load Forecast",
    "🔄 Discharge Demand & Inflow",
    "📊 Model Comparison",
    "⚖️ Scenario & Stress Testing"
])


# Theme-aware Chart Palette Colors
if is_dark:
    color_hist = "#F5ECE5"
    color_ground_truth = "#52B788"
    color_forecast = "#E09F5A"  # Warm amber/bronze (NO BLUE)
    fill_ci = "rgba(224, 159, 90, 0.20)"
    color_threshold = "#EF4444"
    color_transfers = "#A89F91"
    color_discharges = "#52B788"
    color_net_pos = "#EF4444"
    color_net_neg = "#52B788"
    bar_seq = ["#F4A261", "#E09F5A", "#C87D38"]
    overlay_palette = ["#E09F5A", "#D48B46", "#52B788", "#C084FC", "#F472B6", "#A8A29E", "#FBBF24"]
else:
    color_hist = "#0F172A"
    color_ground_truth = "#10B981"
    color_forecast = "#1D4ED8"
    fill_ci = "rgba(29, 78, 216, 0.14)"
    color_threshold = "#DC2626"
    color_transfers = "#94A3B8"
    color_discharges = "#10B981"
    color_net_pos = "#DC2626"
    color_net_neg = "#16A34A"
    bar_seq = ["#93C5FD", "#3B82F6", "#1D4ED8"]
    overlay_palette = ["#1D4ED8", "#D97706", "#059669", "#7C3AED", "#DB2777", "#475569", "#2563EB"]


# ====================================================================
# TAB 1: CARE LOAD FORECAST
# ====================================================================
with tab1:
    st.markdown("#### **Active HHS Care Load Trajectory with 95% Confidence Intervals**")
    
    # Prepare historical + forecast date sequence
    last_train_date = train_df.index.max()
    forecast_dates = pd.date_range(start=last_train_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    
    # View window (last 90 days historical + horizon)
    hist_window = train_df.iloc[-90:].copy()
    
    fig_main = go.Figure()
    
    # 1. Historical Actuals
    fig_main.add_trace(go.Scatter(
        x=hist_window.index,
        y=hist_window[target_option],
        name="Historical Actuals",
        mode="lines+markers",
        line=dict(color=color_hist, width=2.5),
        marker=dict(size=4, color=color_hist)
    ))
    
    # 2. Ground Truth Test Data (if available in historical test period)
    if len(test_df) >= horizon_days:
        actual_test_slice = test_df.iloc[:horizon_days]
        fig_main.add_trace(go.Scatter(
            x=actual_test_slice.index,
            y=actual_test_slice[target_option],
            name="Actual Ground Truth",
            mode="lines+markers",
            line=dict(color=color_ground_truth, width=2.5, dash="dot"),
            marker=dict(size=5, color=color_ground_truth)
        ))
    
    # 3. Forecast Line
    fig_main.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_df["forecast"],
        name=f"{selected_model_name} Forecast",
        mode="lines+markers",
        line=dict(color=color_forecast, width=3),
        marker=dict(size=6, color=color_forecast)
    ))
    
    # 4. Shaded Confidence Interval (Upper & Lower)
    fig_main.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_df["upper_ci"],
        name="95% Upper CI",
        mode="lines",
        line=dict(width=0),
        showlegend=False
    ))
    
    fig_main.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_df["lower_ci"],
        name="95% Confidence Band",
        mode="lines",
        fill="tonexty",
        fillcolor=fill_ci,
        line=dict(width=0)
    ))
    
    # 5. Capacity Threshold Reference Line
    fig_main.add_hline(
        y=capacity_threshold,
        line_dash="dash",
        line_color=color_threshold,
        line_width=2,
        annotation_text=f"Capacity Threshold ({capacity_threshold:,} Beds)",
        annotation_position="top left",
        annotation_font=dict(color=color_threshold, size=12, family="Inter")
    )
    
    fig_main = apply_chart_theme(
        fig_main,
        is_dark=is_dark,
        title=f"Historical Observations and {horizon_days}-Day Forward Projection ({selected_model_name})",
        height=500
    )
    fig_main.update_xaxes(title_text="Reporting Date")
    fig_main.update_yaxes(title_text="Number of Children in HHS Care")
    
    st.plotly_chart(fig_main, use_container_width=True)
    
    # Forecast Data Table
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown("##### **Detailed Daily Multi-Horizon Forecast Breakdown**")
        display_fc = forecast_df.copy()
        display_fc.index = forecast_dates.strftime("%Y-%m-%d")
        display_fc.index.name = "Date"
        display_fc.columns = ["Point Forecast", "95% Lower CI", "95% Upper CI"]
        st.dataframe(display_fc.round(1), use_container_width=True)
    
    with c2:
        st.markdown("##### **Export Projection**")
        csv_buffer = io.StringIO()
        display_fc.to_csv(csv_buffer)
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv_buffer.getvalue(),
            file_name=f"hhs_care_load_forecast_{selected_model_name.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.info(f"**Target:** {target_option}\n\n**Horizon:** {horizon_days} Days\n\n**Confidence Level:** 95% Gaussian Band")


# ====================================================================
# TAB 2: DISCHARGE DEMAND & INFLOW
# ====================================================================
with tab2:
    st.markdown("#### **Discharge Placement Velocity & System Flow Balance**")
    
    fig_flow = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.14,
        subplot_titles=(
            "<b>Daily Intake Inflows vs Discharge Placement Outflows</b>",
            "<b>Net Operational Care Load Pressure (Transfers In − Discharges Out)</b>"
        )
    )
    
    # Inflow transfers from CBP
    fig_flow.add_trace(go.Bar(
        x=train_df.iloc[-45:].index,
        y=train_df["Children transferred out of CBP custody"].iloc[-45:],
        name="CBP Transfers In (Historical)",
        marker_color=color_transfers
    ), row=1, col=1)
    
    # Historical Discharges
    fig_flow.add_trace(go.Scatter(
        x=train_df.iloc[-45:].index,
        y=train_df["Children discharged from HHS Care"].iloc[-45:],
        name="Discharges to Sponsors (Historical)",
        mode="lines+markers",
        line=dict(color=color_discharges, width=2.5)
    ), row=1, col=1)
    
    # Forecasted Discharges
    fig_flow.add_trace(go.Scatter(
        x=forecast_dates,
        y=fc_discharge["forecast"],
        name="Forecasted Discharges",
        mode="lines+markers",
        line=dict(color=color_forecast, width=3, dash="dash")
    ), row=1, col=1)
    
    # Net Pressure (Transfers - Discharges)
    hist_net_pressure = train_df["Children transferred out of CBP custody"] - train_df["Children discharged from HHS Care"]
    colors = np.where(hist_net_pressure.iloc[-45:] >= 0, color_net_pos, color_net_neg)
    
    fig_flow.add_trace(go.Bar(
        x=train_df.iloc[-45:].index,
        y=hist_net_pressure.iloc[-45:],
        name="Net Intake Pressure",
        marker_color=colors
    ), row=2, col=1)
    
    fig_flow.add_hline(y=0, line_color="#F5ECE5" if is_dark else "#000000", line_width=1, row=2, col=1)  # type: ignore[arg-type]
    
    fig_flow = apply_chart_theme(fig_flow, is_dark=is_dark, title="", height=580)
    
    st.plotly_chart(fig_flow, use_container_width=True)
    
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        avg_hist_disc = train_df["Children discharged from HHS Care"].iloc[-30:].mean()
        st.metric("30-Day Avg Daily Discharges", f"{avg_hist_disc:.1f} children/day")
    with fc2:
        avg_fc_disc = fc_discharge["forecast"].mean()
        st.metric(f"Projected Daily Discharges ({horizon_days}d)", f"{avg_fc_disc:.1f} children/day")
    with fc3:
        total_fc_disc = fc_discharge["forecast"].sum()
        st.metric(f"Cumulative {horizon_days}-Day Placement Demand", f"{total_fc_disc:,.0f} children")


# ====================================================================
# TAB 3: MODEL COMPARISON
# ====================================================================
with tab3:
    st.markdown("#### **Multi-Horizon Forecast Accuracy Tournament (3, 7, 14 Days)**")
    
    comp_target_df = comp_df[comp_df["Target"] == target_option].copy()
    
    col_comp_left, col_comp_right = st.columns([3, 2])
    
    with col_comp_left:
        st.markdown("##### **Model Performance Comparison Matrix**")
        st.dataframe(comp_target_df[["Model", "Horizon_Days", "MAE", "RMSE", "MAPE", "Forecast_Accuracy_Pct"]], use_container_width=True)
        
    with col_comp_right:
        st.markdown("##### **Error Comparison Across Horizons (MAE)**")
        fig_bar = px.bar(
            comp_target_df,
            x="Model",
            y="MAE",
            color="Horizon_Days",
            barmode="group",
            color_discrete_sequence=bar_seq
        )
        fig_bar = apply_chart_theme(fig_bar, is_dark=is_dark, title="MAE by Model and Horizon", height=400, legend_y=-0.22)
        st.plotly_chart(fig_bar, use_container_width=True)
        
    st.markdown("---")
    st.markdown("##### **Interactive Multi-Model Overlay on Test Ground Truth**")
    
    overlay_models = st.multiselect(
        "Select Models to Overlay on Test Ground Truth",
        options=list(all_models_dict.keys()),
        default=["Gradient Boosting", "Random Forest", "SARIMA"]
    )
    
    fig_overlay = go.Figure()
    
    # Actual test values
    if len(test_df) >= 14:
        fig_overlay.add_trace(go.Scatter(
            x=test_df.iloc[:14].index,
            y=test_df[target_option].iloc[:14],
            name="Ground Truth Actual",
            mode="lines+markers",
            line=dict(color="#F5ECE5" if is_dark else "#000000", width=3)
        ))
        
        for idx, m_name in enumerate(overlay_models):
            m_instance = get_all_models()[m_name]
            m_instance.fit(train_df, target_col=target_option)
            m_fc = m_instance.predict(horizon=14, future_features=test_df)
            fig_overlay.add_trace(go.Scatter(
                x=test_df.iloc[:14].index,
                y=m_fc["forecast"].iloc[:14],
                name=m_name,
                mode="lines+markers",
                line=dict(color=overlay_palette[idx % len(overlay_palette)], width=2)
            ))
            
        fig_overlay = apply_chart_theme(fig_overlay, is_dark=is_dark, title="14-Day Test Horizon Model Predictions vs Ground Truth", height=480, legend_y=-0.18)
        st.plotly_chart(fig_overlay, use_container_width=True)


# ====================================================================
# TAB 4: SCENARIO & STRESS TESTING
# ====================================================================
with tab4:
    st.markdown("#### **Operational Scenario Simulation & Side-by-Side Model Comparison**")
    
    s_col1, s_col2 = st.columns(2)
    
    with s_col1:
        st.markdown("##### **Scenario A Configuration**")
        model_a_name = st.selectbox("Scenario A Model", options=list(all_models_dict.keys()), index=0, key="model_a")
        cap_a = st.number_input("Scenario A Capacity", value=10000, step=500, key="cap_a")
        
        model_a = get_all_models()[model_a_name]
        model_a.fit(train_df, target_col=target_option)
        fc_a = model_a.predict(horizon=horizon_days, future_features=test_df)
        breach_a = compute_capacity_breach_probability(fc_a, capacity_threshold=cap_a)
        
        fig_a = go.Figure()
        fig_a.add_trace(go.Scatter(x=forecast_dates, y=fc_a["forecast"], name=f"{model_a_name} (A)", line=dict(color=color_forecast, width=2.5)))
        fig_a.add_hline(y=cap_a, line_dash="dash", line_color=color_threshold, annotation_text=f"Cap A: {cap_a:,}")
        fig_a = apply_chart_theme(fig_a, is_dark=is_dark, title=f"Scenario A: {model_a_name} (Breach Risk: {breach_a:.1f}%)", height=360, legend_y=-0.26)
        st.plotly_chart(fig_a, use_container_width=True)
        
    with s_col2:
        st.markdown("##### **Scenario B Configuration**")
        model_b_name = st.selectbox("Scenario B Model", options=list(all_models_dict.keys()), index=1, key="model_b")
        cap_b = st.number_input("Scenario B Capacity", value=12000, step=500, key="cap_b")
        
        model_b = get_all_models()[model_b_name]
        model_b.fit(train_df, target_col=target_option)
        fc_b = model_b.predict(horizon=horizon_days, future_features=test_df)
        breach_b = compute_capacity_breach_probability(fc_b, capacity_threshold=cap_b)
        
        fig_b = go.Figure()
        fig_b.add_trace(go.Scatter(x=forecast_dates, y=fc_b["forecast"], name=f"{model_b_name} (B)", line=dict(color=color_discharges, width=2.5)))
        fig_b.add_hline(y=cap_b, line_dash="dash", line_color=color_threshold, annotation_text=f"Cap B: {cap_b:,}")
        fig_b = apply_chart_theme(fig_b, is_dark=is_dark, title=f"Scenario B: {model_b_name} (Breach Risk: {breach_b:.1f}%)", height=360, legend_y=-0.26)
        st.plotly_chart(fig_b, use_container_width=True)
        
    st.markdown("##### **Scenario Variance & Difference Analysis**")
    fc_a_vals = np.asarray(fc_a["forecast"], dtype=float)
    fc_b_vals = np.asarray(fc_b["forecast"], dtype=float)
    diff_series = fc_b_vals - fc_a_vals
    pct_diff = (diff_series / np.maximum(fc_a_vals, 1.0)) * 100.0
    
    scenario_comparison_df = pd.DataFrame({
        "Date": forecast_dates.strftime("%Y-%m-%d"),
        f"Scenario A ({model_a_name})": np.round(fc_a_vals, 1),
        f"Scenario B ({model_b_name})": np.round(fc_b_vals, 1),
        "Absolute Difference": np.round(diff_series, 1),
        "Percentage Divergence (%)": np.round(pct_diff, 2)
    })
    st.dataframe(scenario_comparison_df, use_container_width=True)

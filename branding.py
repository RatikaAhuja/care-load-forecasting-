"""
branding.py - CareCast Visual Branding & Reusable Header Component

Provides the official CareCast SVG logo, header rendering function,
and theme styling helpers for the Streamlit dashboard supporting
both Light and Warm Espresso / Brown Dark Mode.
"""

import os
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(PROJECT_ROOT, "assets", "carecast_logo.svg")

# Standalone inline SVG fallbacks
LIGHT_SVG_LOGO = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100%" height="100%" fill="none">
  <!-- House / Shelter Outline (#000000) -->
  <path d="M 16 46 L 50 18 L 84 46" stroke="#000000" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M 25 44 L 25 82 L 75 82 L 75 44" stroke="#000000" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M 43 82 L 43 64 L 57 64 L 57 82" stroke="#000000" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>

  <!-- Upward Trend Line / Forecast (#2f6fed) Breaking Through -->
  <path d="M 12 72 L 34 60 L 54 44 L 88 15" stroke="#2f6fed" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
  
  <!-- Arrowhead on Trend Line -->
  <path d="M 72 15 L 88 15 L 88 31" stroke="#2f6fed" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round"/>

  <!-- Data Points along Trend Line -->
  <circle cx="12" cy="72" r="4.5" fill="#2f6fed" stroke="#FFFFFF" stroke-width="1.5"/>
  <circle cx="34" cy="60" r="4.5" fill="#2f6fed" stroke="#FFFFFF" stroke-width="1.5"/>
  <circle cx="54" cy="44" r="4.5" fill="#2f6fed" stroke="#FFFFFF" stroke-width="1.5"/>
  <circle cx="88" cy="15" r="5.5" fill="#2f6fed" stroke="#FFFFFF" stroke-width="1.5"/>
</svg>"""

DARK_BROWN_SVG_LOGO = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100%" height="100%" fill="none">
  <!-- House / Shelter Outline (Warm Cream #F5ECE5) -->
  <path d="M 16 46 L 50 18 L 84 46" stroke="#F5ECE5" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M 25 44 L 25 82 L 75 82 L 75 44" stroke="#F5ECE5" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M 43 82 L 43 64 L 57 64 L 57 82" stroke="#F5ECE5" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>

  <!-- Upward Trend Line / Forecast (Warm Bronze/Amber #E09F5A - NO BLUE) -->
  <path d="M 12 72 L 34 60 L 54 44 L 88 15" stroke="#E09F5A" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
  
  <!-- Arrowhead on Trend Line -->
  <path d="M 72 15 L 88 15 L 88 31" stroke="#E09F5A" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round"/>

  <!-- Data Points along Trend Line -->
  <circle cx="12" cy="72" r="4.5" fill="#E09F5A" stroke="#1C1613" stroke-width="1.5"/>
  <circle cx="34" cy="60" r="4.5" fill="#E09F5A" stroke="#1C1613" stroke-width="1.5"/>
  <circle cx="54" cy="44" r="4.5" fill="#E09F5A" stroke="#1C1613" stroke-width="1.5"/>
  <circle cx="88" cy="15" r="5.5" fill="#E09F5A" stroke="#1C1613" stroke-width="1.5"/>
</svg>"""


def get_logo_svg(is_dark: bool = False, *args, **kwargs) -> str:
    """
    Returns the SVG logo matching the active theme.
    In dark mode, uses warm cream house outline with warm amber/bronze trendline.
    In light mode, uses black house outline with blue trendline.
    """
    # Safety check for keyword or positional arguments
    if not is_dark and kwargs.get("is_dark"):
        is_dark = bool(kwargs["is_dark"])
    elif args and isinstance(args[0], bool):
        is_dark = bool(args[0])

    if is_dark:
        return DARK_BROWN_SVG_LOGO
    
    if os.path.exists(LOGO_PATH):
        try:
            with open(LOGO_PATH, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return LIGHT_SVG_LOGO
    return LIGHT_SVG_LOGO


def render_header(is_dark: bool = False, *args, **kwargs) -> None:
    """
    Renders the CareCast unified brand header at the top of the Streamlit app.
    
    Layout:
    - Left: CareCast SVG logo icon (~52x52px)
    - Next to it: 'CareCast' app title
    - Below title: 'Forecasting capacity before crisis strikes' tagline
    - Bottom: Horizontal divider line
    
    Dynamically supports Light (#FFFFFF / #000000) and Warm Espresso Dark Mode (#1C1613 / #F5ECE5).
    """
    # Safety check for keyword or positional arguments
    if not is_dark and kwargs.get("is_dark"):
        is_dark = bool(kwargs["is_dark"])
    elif args and isinstance(args[0], bool):
        is_dark = bool(args[0])

    svg_content = get_logo_svg(is_dark=is_dark)

    bg_color = "#1C1613" if is_dark else "transparent"
    title_color = "#F5ECE5" if is_dark else "#0F172A"
    tagline_color = "#C7B5A7" if is_dark else "#334155"
    divider_color = "#44352D" if is_dark else "#BFDBFE"

    header_html = f"""
    <div style="
        display: flex;
        align-items: center;
        gap: 16px;
        background-color: {bg_color} !important;
        padding: 4px 0px 10px 0px;
        margin-top: -15px;
        margin-bottom: 6px;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    ">
        <div style="
            width: 52px;
            height: 52px;
            min-width: 52px;
            display: flex;
            align-items: center;
            justify-content: center;
        ">
            {svg_content}
        </div>
        <div style="display: flex; flex-direction: column; justify-content: center;">
            <div style="
                font-size: 1.85rem;
                font-weight: 800;
                line-height: 1.15;
                color: {title_color} !important;
                letter-spacing: -0.02em;
                margin: 0;
                padding: 0;
            ">CareCast</div>
            <div style="
                font-size: 0.92rem;
                font-weight: 500;
                color: {tagline_color} !important;
                margin-top: 2px;
                padding: 0;
            ">Forecasting capacity before crisis strikes</div>
        </div>
    </div>
    <div style="
        width: 100%;
        height: 1px;
        background-color: {divider_color};
        margin-bottom: 16px;
        border: none;
    "></div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

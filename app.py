"""
Equity-Commodities Spillover Monitor
Purdue University · Daniels School of Business

Launch:
    streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import date

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).parent))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Equity-Commodities Spillover | Purdue Daniels",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
# (Sidebar fully hidden; content pushed below the fixed 56 px navbar iframe)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --font-sans: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
    --gold:       #CFB991;
    --black:      #000000;
    --ink-soft:   #333333;
    --ink-muted:  #444444;
    --ink-faint:  #666666;
    --bg:         #ffffff;
    --bg-warm:    #fafaf8;
    --rule:       #E8E5E0;
    --red:        #c0392b;
    --green:      #2e7d32;
}

* { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    font-family: var(--font-sans);
    background: var(--bg);
    color: var(--black);
}

/* ── Hide Streamlit chrome + sidebar ── */
#MainMenu, footer,
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stHeader"],
[data-testid="stSidebar"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[aria-label="Close sidebar"],
button[aria-label="Open sidebar"] {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
}

/* ── Content area — padded below the 72 px fixed navbar ── */
.main .block-container {
    padding-top:    86px !important;
    padding-bottom: 0 !important;
    padding-left:   2.2rem !important;
    padding-right:  2.2rem !important;
    max-width: 1360px;
}
/* Kill any Streamlit-injected bottom chrome that creates white space */
[data-testid="stBottom"], .reportview-container .main footer { display: none !important; }

/* ── Headings ── */
h1 { font-size: 1.25rem !important; font-weight: 700 !important; letter-spacing: -0.01em; }
h2 { font-size: 0.95rem !important; font-weight: 700 !important; margin-top: 1.4rem !important; }
h3 { font-size: 0.82rem !important; font-weight: 600 !important; margin-top: 1rem !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--bg-warm); border: 1px solid var(--rule);
    border-radius: 4px; padding: 0.6rem 0.8rem; transition: box-shadow 0.2s;
}
[data-testid="metric-container"]:hover { box-shadow: 0 2px 12px rgba(0,0,0,0.06); border-color: var(--gold); }
[data-testid="stMetricLabel"] {
    font-size: 0.58rem !important; font-weight: 600 !important;
    letter-spacing: 0.14em !important; text-transform: uppercase !important; color: #555960 !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--font-mono) !important; font-size: 1.05rem !important; font-weight: 700 !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] { border-bottom: 1px solid var(--rule); gap: 0; }
[data-testid="stTabs"] button[role="tab"] {
    font-size: 0.7rem !important; font-weight: 500 !important;
    letter-spacing: 0.04em !important; text-transform: uppercase !important;
    color: var(--ink-muted) !important; padding: 0.5rem 1rem !important;
    border-bottom: 2px solid transparent; transition: all 0.2s;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: var(--black) !important; border-bottom-color: var(--gold) !important; font-weight: 700 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--black) !important; color: var(--gold) !important;
    border: 1px solid var(--black) !important; border-radius: 3px !important;
    font-size: 0.68rem !important; font-weight: 600 !important;
    letter-spacing: 0.06em !important; text-transform: uppercase !important;
    padding: 0.45rem 1.4rem !important; transition: all 0.2s !important;
}
.stButton > button:hover { background: var(--gold) !important; color: var(--black) !important; border-color: var(--gold) !important; }

/* ── Form controls ── */
[data-testid="stSelectbox"] label, [data-testid="stMultiSelect"] label,
[data-testid="stSlider"] label, .stCheckbox label span {
    font-size: 0.65rem !important; font-weight: 600 !important;
    letter-spacing: 0.08em !important; text-transform: uppercase !important; color: var(--ink-soft) !important;
}

/* ── Date inputs in the control strip ── */
[data-testid="stDateInput"] label {
    font-size: 0.58rem !important; font-weight: 700 !important;
    letter-spacing: 0.14em !important; text-transform: uppercase !important; color: var(--ink-faint) !important;
}
[data-testid="stDateInput"] [data-baseweb="input"] {
    background: var(--bg-warm) !important; border: 1px solid var(--rule) !important;
    border-radius: 3px !important; font-size: 0.70rem !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid var(--rule); border-radius: 4px; }
[data-testid="stDataFrame"] th {
    background: var(--bg-warm) !important; font-size: 0.65rem !important;
    font-weight: 600 !important; letter-spacing: 0.08em !important;
    text-transform: uppercase !important; color: var(--ink-soft) !important;
}
[data-testid="stDataFrame"] td { font-family: var(--font-mono) !important; font-size: 0.7rem !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] { color: var(--gold) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #DEDAD5; border-radius: 3px; }

/* ── Divider ── */
hr { border: none; border-top: 1px solid var(--rule); margin: 1.2rem 0; }

/* ── Fade-in ── */
@keyframes fadeInUp { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
.main .block-container > div { animation: fadeInUp 0.28s ease; }

/* ═══════════════════════════════════════════════════════════════════════════
   FORMALIZATION — Tighter, institutional-grade UI polish
   ═══════════════════════════════════════════════════════════════════════════ */

/* ── Streamlit column gap tightening ── */
[data-testid="stHorizontalBlock"] { gap: 1.2rem !important; align-items: stretch !important; }
[data-testid="stColumn"] { padding: 0 !important; }

/* ── Dividers rendered by st.markdown('---') ── */
[data-testid="stMarkdownContainer"] hr { margin: 0.6rem 0 !important; }

/* ── Spinner font ── */
[data-testid="stSpinner"] p { font-size: 0.68rem !important; color: var(--ink-faint) !important; letter-spacing: 0.06em !important; }

/* ── Alert boxes ── */
[data-testid="stAlert"] {
    border-radius: 3px !important;
    font-size: 0.72rem !important;
    border-left-width: 3px !important;
    padding: 0.55rem 0.9rem !important;
}

/* ── Expander header ── */
[data-testid="stExpander"] summary {
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--ink-muted) !important;
}
[data-testid="stExpander"] summary:hover { color: var(--black) !important; }
[data-testid="stExpander"] { border-radius: 3px !important; border-color: var(--rule) !important; }

/* ── Multiselect tag pill ── */
[data-baseweb="tag"] {
    border-radius: 3px !important;
    font-size: 0.60rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    background: rgba(207,185,145,0.12) !important;
    border-color: rgba(207,185,145,0.30) !important;
    color: #8E6F3E !important;
}

/* ── Select dropdown menu ── */
[data-baseweb="popover"] {
    border-radius: 3px !important;
    border: 1px solid var(--rule) !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.10) !important;
}
[role="option"] {
    font-size: 0.70rem !important;
    padding: 0.45rem 0.85rem !important;
}
[role="option"][aria-selected="true"] { color: var(--gold) !important; }

/* ── Slider ── */
[data-testid="stSlider"] [role="slider"] { background: var(--gold) !important; border-color: var(--gold) !important; }
[data-testid="stSlider"] div[data-testid="stThumbValue"] { font-size: 0.62rem !important; }

/* ── Date inputs ── */
[data-testid="stDateInput"] input { font-size: 0.68rem !important; font-family: var(--font-mono) !important; }

/* ── Plotly chart container borders ── */
.stPlotlyChart { border: 1px solid var(--rule) !important; border-radius: 4px !important; overflow: hidden; }

/* ── KPI number in stMetric — enforce mono font ── */
[data-testid="stMetricValue"] > div { font-family: var(--font-mono) !important; }

/* ── Primary button — refined ── */
.stButton > button[kind="primary"] {
    background: var(--gold) !important;
    color: #000 !important;
    border-color: var(--gold) !important;
    font-weight: 700 !important;
}

/* ── Selectbox & slider label uppercase ── */
[data-testid="stSelectbox"] > label p,
[data-testid="stMultiSelect"] > label p,
[data-testid="stSlider"] > label p {
    font-size: 0.58rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: var(--ink-faint) !important;
}

/* ── Column content vertical alignment ── */
.stPlotlyChart { margin-bottom: 0 !important; }

/* ── Section headings via st.subheader ── */
[data-testid="stHeading"] h2 {
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.10em !important;
    text-transform: uppercase !important;
    color: var(--black) !important;
    border-bottom: 1px solid var(--rule) !important;
    padding-bottom: 0.35rem !important;
    margin-bottom: 0.8rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Dark mode CSS (injected by JS from navbar iframe into parent document) ────
_DM_CSS = """
/* ════════════════════════════════════════════════════════════════════════════
   DARK MODE — Equity & Commodities Spillover Dashboard
   All inline-style overrides use !important which wins over non-!important
   inline styles per CSS cascade spec.
   ════════════════════════════════════════════════════════════════════════════ */

/* ── Page background ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
section[data-testid="stMain"],
.main,
.main .block-container,
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
[data-testid="stColumn"] {
    background: #0f1117 !important;
    color: #e8e9ed !important;
}

/* ── Headings & generic text ── */
h1, h2, h3, label, p, span, div { color: inherit; }
[data-testid="stMarkdownContainer"] * { color: #e8e9ed; }
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 { color: #CFB991 !important; }
h1, h2, h3, label, p { color: #e8e9ed !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #1a1d27 !important;
    border-color: #2a2d3a !important;
}
[data-testid="stMetricLabel"]  { color: #8890a1 !important; }
[data-testid="stMetricValue"]  { color: #e8e9ed !important; }
[data-testid="stMetricDelta"]  { color: #CFB991 !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    border-bottom-color: #2a2d3a !important;
    background: #0f1117 !important;
}
[data-testid="stTabs"] button[role="tab"]                    { color: #8890a1 !important; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color: #e8e9ed !important; }

/* ── Buttons ── */
.stButton > button {
    background: #1a1d27 !important;
    color: #CFB991 !important;
    border-color: #CFB991 !important;
}
.stButton > button:hover {
    background: #CFB991 !important;
    color: #000 !important;
}

/* ── Select / multiselect / input ── */
[data-baseweb="select"] > div,
[data-baseweb="input"]  > div  { background: #1a1d27 !important; border-color: #2a2d3a !important; }
[data-baseweb="select"] input,
[data-baseweb="input"]  input  { color: #e8e9ed !important; }
[data-baseweb="popover"] > div { background: #1a1d27 !important; }
[role="option"]                { background: #1a1d27 !important; color: #e8e9ed !important; }
[role="option"]:hover          { background: #252a3a !important; }
[data-baseweb="tag"]           { background: #252a3a !important; color: #CFB991 !important; }
[data-baseweb="tag"] span      { color: #CFB991 !important; }
[data-testid="stSelectbox"] label,
[data-testid="stMultiSelect"] label,
[data-testid="stSlider"] label        { color: #8890a1 !important; }
[data-testid="stDateInput"] label     { color: #8890a1 !important; }
[data-testid="stDateInput"] [data-baseweb="input"] {
    background: #1a1d27 !important;
    border-color: #2a2d3a !important;
    color: #e8e9ed !important;
}

/* ── Expanders ── */
[data-testid="stExpander"]         { border-color: #2a2d3a !important; background: #1a1d27 !important; }
[data-testid="stExpander"] summary { color: #b8bec8 !important; }
[data-testid="stExpander"] > div   { background: #1a1d27 !important; }

/* ── DataFrames ── */
[data-testid="stDataFrame"]       { border-color: #2a2d3a !important; background: #0f1117 !important; }
[data-testid="stDataFrame"] thead th { background: #1a1d27 !important; color: #8890a1 !important; }
[data-testid="stDataFrame"] tbody td { color: #e8e9ed !important; }
[data-testid="stDataFrame"] div    { background: #0f1117 !important; color: #e8e9ed !important; }

/* ── Alerts ── */
[data-testid="stAlert"]   { background: #1a1d27 !important; border-color: #2a2d3a !important; color: #e8e9ed !important; }
[data-testid="stInfo"]    { color: #e8e9ed !important; }
[data-testid="stWarning"] { color: #e8e9ed !important; }
[data-testid="stError"]   { color: #e8e9ed !important; }

/* ── Dividers / scrollbar ── */
hr { border-top-color: #2a2d3a !important; }
::-webkit-scrollbar-thumb { background: #2a2d3a !important; }

/* ══════════════════════════════════════════════════════════════════════════
   INLINE-STYLE OVERRIDES  (cascade: stylesheet !important > inline)
   ══════════════════════════════════════════════════════════════════════════ */

/* -- Backgrounds -- */
[style*="background:#fafaf8"],   [style*="background: #fafaf8"]   { background: #1a1d27 !important; }
[style*="background:#fff"],      [style*="background: #fff"]      { background: #1e2130 !important; }
[style*="background:#fffdf5"],   [style*="background: #fffdf5"]   { background: #1a1f2e !important; }
[style*="background:#f0ede8"],   [style*="background: #f0ede8"]   { background: #1c1e2a !important; }
[style*="background:rgba(207,185,145,0.10)"],
[style*="background: rgba(207,185,145,0.10)"] { background: rgba(207,185,145,0.15) !important; }

/* -- Text -- */
[style*="color:#000000"],  [style*="color: #000000"]  { color: #e8e9ed !important; }
[style*="color:#000;"],    [style*="color: #000;"]    { color: #e8e9ed !important; }
[style*="color:#000\""],   [style*="color: #000\""]   { color: #e8e9ed !important; }
[style*="color:#111111"],  [style*="color: #111111"]  { color: #e8e9ed !important; }
[style*="color:#333333"],  [style*="color: #333333"]  { color: #d1d5db !important; }
[style*="color:#333;"],    [style*="color: #333;"]    { color: #d1d5db !important; }
[style*="color:#333\""],   [style*="color: #333\""]   { color: #d1d5db !important; }
[style*="color:#444444"],  [style*="color: #444444"]  { color: #b8bec8 !important; }
[style*="color:#444;"],    [style*="color: #444;"]    { color: #b8bec8 !important; }
[style*="color:#555960"],  [style*="color: #555960"]  { color: #8890a1 !important; }
[style*="color:#555;"],    [style*="color: #555;"]    { color: #8890a1 !important; }
[style*="color:#666666"],  [style*="color: #666666"]  { color: #8890a1 !important; }
[style*="color:#666;"],    [style*="color: #666;"]    { color: #8890a1 !important; }
[style*="color:#666\""],   [style*="color: #666\""]   { color: #8890a1 !important; }
[style*="color:#777"],     [style*="color: #777"]     { color: #6b7280 !important; }
[style*="color:#888"],     [style*="color: #888"],
[style*="color: #888"]                                { color: #6b7280 !important; }

/* -- Borders -- */
[style*="border:1px solid #E8E5E0"],  [style*="border: 1px solid #E8E5E0"]  { border-color: #2a2d3a !important; }
[style*="border-top:1px solid #E8E5E0"], [style*="border-top: 1px solid #E8E5E0"] { border-top-color: #2a2d3a !important; }
[style*="border-bottom:1px solid #E8E5E0"] { border-bottom-color: #2a2d3a !important; }
[style*="border-top:3px solid"],      [style*="border-top: 3px solid"]      { border-top-color: #CFB991 !important; }

/* -- Definition block header (black bg → very dark) -- */
[style*="background:#000000"], [style*="background: #000000"],
[style*="background:#000;"],   [style*="background: #000;"]   { background: #050608 !important; }

/* ── Text inputs & text areas ── */
[data-testid="stTextInput"] > div,
[data-testid="stTextInput"] > div > div,
[data-baseweb="base-input"] {
    background: #1a1d27 !important;
    border-color: #2a2d3a !important;
}
[data-testid="stTextInput"] input,
[data-testid="stTextArea"]  textarea {
    background: #1a1d27 !important;
    color:       #e8e9ed !important;
    border-color: #2a2d3a !important;
    caret-color: #CFB991 !important;
}
[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"]  textarea::placeholder { color: #4a5060 !important; }
[data-testid="stTextInput"] label,
[data-testid="stTextArea"]  label { color: #8890a1 !important; }

/* ── Text area outer wrapper ── */
[data-testid="stTextArea"] > div {
    background: #1a1d27 !important;
    border-color: #2a2d3a !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInputContainer"],
[data-testid="stChatInputContainer"] > div {
    background: #1a1d27 !important;
    border-color: #2a2d3a !important;
}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInputContainer"] textarea {
    background: #1a1d27 !important;
    color:       #e8e9ed !important;
    caret-color: #CFB991 !important;
}
[data-testid="stChatInput"] textarea::placeholder,
[data-testid="stChatInputContainer"] textarea::placeholder { color: #4a5060 !important; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: #1a1d27 !important;
    border-color: #2a2d3a !important;
}

/* ── Slider track & thumb ── */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #CFB991 !important;
    border-color: #CFB991 !important;
}
[data-testid="stSlider"] div[data-testid="stTickBar"] { color: #6b7280 !important; }

/* ── Number input ── */
[data-testid="stNumberInput"] input {
    background: #1a1d27 !important;
    color: #e8e9ed !important;
    border-color: #2a2d3a !important;
}
[data-testid="stNumberInput"] > div { background: #1a1d27 !important; border-color: #2a2d3a !important; }

/* ── Code blocks ── */
[data-testid="stCode"],
[data-testid="stCode"] pre,
code, pre {
    background: #0d0f18 !important;
    color: #c8cdd8 !important;
    border-color: #2a2d3a !important;
}

/* ── Spinner / status ── */
[data-testid="stSpinner"] > div { color: #CFB991 !important; }

/* ── Bottom bar (chat input sticky area) ── */
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottom"] section { background: #0f1117 !important; border-top-color: #2a2d3a !important; }

/* ── Plotly chart borders ── */
.stPlotlyChart { border-color: #2a2d3a !important; }

/* ── Any remaining white container patches ── */
[data-testid="stForm"],
[data-testid="stForm"] > div { background: #1a1d27 !important; border-color: #2a2d3a !important; }
iframe { background: transparent !important; }

/* ── Date inputs (all wrappers + input element) ── */
[data-testid="stDateInput"] div,
[data-testid="stDateInput"] input,
[data-testid="stDateInput"] [data-baseweb],
[data-testid="stDateInput"] [role="textbox"],
[data-baseweb="input-container"],
[data-baseweb="calendar"] { background: #1a1d27 !important; color: #e8e9ed !important; border-color: #2a2d3a !important; }

/* ── Radio / checkbox labels ── */
[data-testid="stRadio"]   label { color: #b8bec8 !important; }
[data-testid="stCheckbox"] label { color: #b8bec8 !important; }
[data-testid="stRadio"]   [role="radio"][aria-checked="true"] ~ span,
[data-testid="stRadio"]   label:has([aria-checked="true"]) { color: #CFB991 !important; }

/* ════════════════════════════════════════════════════════════════
   PURDUE GOLD ACCENTS  (dark mode — tasteful, not loud)
   ════════════════════════════════════════════════════════════════ */

/* Page titles */
h1 { color: #CFB991 !important; }
h2 { color: #c4ae88 !important; }
h3 { color: #b8a27a !important; }

/* Streamlit heading widget */
[data-testid="stHeading"] { color: #CFB991 !important; }

/* Metric values */
[data-testid="stMetricValue"] { color: #CFB991 !important; }

/* Active tab — gold underline + text */
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #CFB991 !important;
    border-bottom: 2px solid #CFB991 !important;
}

/* DataTable column headers */
[data-testid="stDataFrame"] thead th {
    background: #1a1d27 !important;
    color: #CFB991 !important;
    border-bottom: 1px solid rgba(207,185,145,0.3) !important;
}

/* Subheader (st.subheader) */
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 { color: #c4ae88 !important; }

/* Dividers (---) → faint gold */
hr { border-top-color: rgba(207,185,145,0.25) !important; }

/* Expander header text */
[data-testid="stExpander"] summary span { color: #CFB991 !important; }

/* Active radio button dot → gold */
[data-baseweb="radio"] [role="radio"][aria-checked="true"] div:first-child {
    border-color: #CFB991 !important;
    background: #CFB991 !important;
}

/* Inline <code> snippets → warm gold tint */
code:not(pre code) { color: #CFB991 !important; background: rgba(207,185,145,0.10) !important; }

"""

# ── API keys ──────────────────────────────────────────────────────────────────
def _get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get("keys", {}).get(key, default) or default
    except Exception:
        return default

_FRED_KEY = _get_secret("fred_api_key")
_FD_KEY   = _get_secret("financial_datasets_key")

# ── Navigation state ──────────────────────────────────────────────────────────
_VALID_PAGES = {
    "overview", "war_impact_map", "geopolitical", "correlation",
    "spillover", "watchlist", "trade_ideas", "stress_test",
    "model_accuracy", "ai_chat", "about_heramb", "about_jiahe", "about_ilian",
}
_ABOUT_PAGES = {"about_heramb", "about_jiahe", "about_ilian"}

_qp = st.query_params.get("page", "")
if _qp in _VALID_PAGES:
    st.session_state["current_page"] = _qp
    st.query_params.clear()
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "overview"

current = st.session_state["current_page"]

# ── Navbar (rendered via components.html — no sanitisation; JS fixes it to top) ─
_fred_dot = "●" if _FRED_KEY else "○"
_fred_col = "#CFB991" if _FRED_KEY else "rgba(255,255,255,0.22)"
_fd_dot   = "●" if _FD_KEY   else "○"
_fd_col   = "#CFB991" if _FD_KEY   else "rgba(255,255,255,0.22)"

_NAVBAR = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
html,body{{height:72px;overflow:visible;background:#000;
  font-family:'DM Sans',-apple-system,BlinkMacSystemFont,sans-serif;
  -webkit-font-smoothing:antialiased}}
#nav{{
  display:flex;align-items:stretch;height:72px;
  padding:0 2rem;background:#000;
  border-bottom:1px solid #1e1e1e;
}}

/* ── Logotype ── */
.brand{{
  display:flex;align-items:center;gap:13px;
  margin-right:2.8rem;flex-shrink:0;
  text-decoration:none;padding:0 2px;
  transition:opacity .15s;
}}
.brand:hover{{opacity:.82}}
/* Gold monogram tile */
.brand-mark{{
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  width:36px;height:36px;flex-shrink:0;
  background:#CFB991;border-radius:5px;
  font-size:.60rem;font-weight:800;
  color:#000;letter-spacing:.06em;line-height:1.15;
  user-select:none;
}}
.brand-mark .mk-top{{font-size:.58rem;font-weight:800;letter-spacing:.10em}}
.brand-mark .mk-bot{{font-size:.44rem;font-weight:600;letter-spacing:.14em;opacity:.70}}
/* Divider */
.brand-div{{width:1px;height:30px;background:rgba(207,185,145,.18);flex-shrink:0}}
/* Text stack */
.brand-text{{display:flex;flex-direction:column;gap:3px}}
.bm{{
  font-size:.82rem;font-weight:700;
  color:#fff;letter-spacing:.005em;
  white-space:nowrap;line-height:1.1;
}}
/* Italic serif-like styling via font-style + lighter weight for the "&" */
.bm em{{font-style:italic;font-weight:400;color:rgba(207,185,145,.85);margin:0 1px}}
.bs{{
  font-size:.50rem;font-weight:500;
  color:rgba(207,185,145,.55);
  letter-spacing:.16em;text-transform:uppercase;
  white-space:nowrap;
}}

/* ── Nav list ── */
ul.links{{display:flex;list-style:none;height:100%;align-items:stretch;margin:0;padding:0;gap:0}}
li.ni{{position:relative;display:flex;align-items:stretch}}

/* Top-level items */
.ni > a.lnk,.ni > span.lnk{{
  display:flex;align-items:center;gap:5px;padding:0 1.05rem;
  font-size:.72rem;font-weight:500;
  color:rgba(255,255,255,.78);
  text-decoration:none;white-space:nowrap;
  border-bottom:2px solid transparent;
  cursor:pointer;user-select:none;
  transition:color .13s,border-color .13s,background .13s;
  letter-spacing:.022em;
}}
.ni > a.lnk:hover,.ni > span.lnk:hover{{
  color:#fff;background:rgba(255,255,255,.04);
  border-bottom-color:rgba(207,185,145,.35);
}}
.ni > a.lnk.active{{color:#CFB991;border-bottom-color:#CFB991;font-weight:600}}
.ni.hd-active > span.lnk{{color:#CFB991;border-bottom-color:#CFB991;font-weight:600}}

/* Caret */
.ct{{font-size:.45rem;opacity:.45;transition:transform .16s;display:inline-block;margin-top:1px}}
.ni:hover .ct{{transform:rotate(180deg);opacity:.75}}

/* ── Dropdown ── */
ul.drop{{
  display:none;position:absolute;top:72px;left:0;
  min-width:214px;background:#070707;
  border:1px solid #282828;border-top:2px solid #CFB991;
  border-radius:0 0 6px 6px;list-style:none;
  margin:0;padding:7px 0;
  box-shadow:0 14px 36px rgba(0,0,0,.70);z-index:99999;
}}
ul.drop li a{{
  display:block;padding:.52rem 1.15rem;
  font-size:.70rem;font-weight:400;
  color:rgba(255,255,255,.55);text-decoration:none;
  border-left:2px solid transparent;white-space:nowrap;
  transition:color .10s,background .10s,border-color .10s;
  letter-spacing:.012em;
}}
ul.drop li a:hover{{color:#fff;background:rgba(255,255,255,.04);border-left-color:rgba(207,185,145,.32)}}
ul.drop li a.active{{color:#CFB991;background:rgba(207,185,145,.07);border-left-color:#CFB991;font-weight:500}}

/* ── Right data-source strip ── */
.ds{{margin-left:auto;display:flex;align-items:center;gap:.85rem;flex-shrink:0;
     font-size:.57rem;letter-spacing:.05em;padding-left:1.4rem}}
.dsi{{display:flex;align-items:center;gap:4px}}
.dsi .dot{{font-size:.50rem}}

/* ── Dark mode toggle ── */
.dm-toggle{{
  display:flex;align-items:center;justify-content:center;
  width:30px;height:30px;
  background:rgba(255,255,255,0.05);
  border:1px solid rgba(207,185,145,0.22);
  border-radius:5px;
  color:rgba(255,255,255,0.65);
  cursor:pointer;
  font-size:0.82rem;
  line-height:1;
  transition:all .15s;
  margin-left:0.5rem;
  flex-shrink:0;
  text-decoration:none;
  user-select:none;
}}
.dm-toggle:hover{{
  background:rgba(207,185,145,0.12);
  border-color:#CFB991;
  color:#CFB991;
}}
.dm-toggle:focus{{outline:none;}}
</style>
</head>
<body>
<script>
(function(){{
  var NAV_H = 72;
  var f = window.frameElement;
  if(!f) return;

  /* ── Hoist iframe to fixed top bar ── */
  function fixNav(){{
    var hs = NAV_H+'px';
    f.style.cssText='position:fixed!important;top:0!important;left:0!important;'+
      'width:100%!important;height:'+hs+'!important;border:none!important;'+
      'z-index:9999!important;margin:0!important;padding:0!important;background:#000!important;';
    var p=f.parentElement;
    if(p) p.style.cssText='position:fixed!important;top:0!important;left:0!important;'+
      'width:100%!important;height:'+hs+'!important;z-index:9999!important;'+
      'margin:0!important;padding:0!important;';
  }}
  fixNav();
  try{{
    new window.parent.MutationObserver(fixNav)
      .observe(window.parent.document.body,{{childList:true,subtree:false,attributes:false}});
  }}catch(e){{}}

  /* ── Parent-document dropdown (bypasses iframe clipping entirely) ── */
  var _timer=null, _pdoc=window.parent.document;

  function cancelTimer(){{ if(_timer){{ clearTimeout(_timer); _timer=null; }} }}

  function removeDropdown(){{
    var el=_pdoc.getElementById('ec-nav-drop');
    if(el) el.remove();
  }}

  function scheduleHide(){{
    cancelTimer();
    _timer=setTimeout(removeDropdown,130);
  }}

  function buildDropdown(li){{
    cancelTimer();
    removeDropdown();

    var items=li.querySelectorAll('ul.drop > li > a');
    if(!items.length) return;

    var ifrR=f.getBoundingClientRect();
    var liR=li.getBoundingClientRect();

    var ul=_pdoc.createElement('ul');
    ul.id='ec-nav-drop';
    ul.style.cssText=
      'position:fixed;top:'+(ifrR.top+NAV_H)+'px;left:'+(ifrR.left+liR.left)+'px;'+
      'min-width:220px;background:#070707;'+
      'border:1px solid #282828;border-top:2px solid #CFB991;'+
      'border-radius:0 0 6px 6px;list-style:none;margin:0;padding:7px 0;'+
      'box-shadow:0 14px 36px rgba(0,0,0,.75);z-index:99999;'+
      'font-family:DM Sans,-apple-system,sans-serif;';

    items.forEach(function(origA){{
      var pg=origA.getAttribute('data-pg');
      var isActive=origA.classList.contains('active');
      var li2=_pdoc.createElement('li');
      var a=_pdoc.createElement('a');
      a.textContent=origA.textContent;
      a.href='?page='+pg;
      a.style.cssText=
        'display:block;padding:.52rem 1.15rem;font-size:.72rem;font-weight:400;'+
        'color:'+(isActive?'#CFB991':'rgba(255,255,255,.72)')+';text-decoration:none;'+
        'border-left:2px solid '+(isActive?'#CFB991':'transparent')+';'+
        'white-space:nowrap;letter-spacing:.012em;'+
        'background:'+(isActive?'rgba(207,185,145,.07)':'')+';'+
        'cursor:pointer;';
      a.addEventListener('mouseover',function(){{
        a.style.color='#fff';
        a.style.background='rgba(255,255,255,.04)';
        a.style.borderLeftColor='rgba(207,185,145,.35)';
      }});
      a.addEventListener('mouseout',function(){{
        a.style.color=isActive?'#CFB991':'rgba(255,255,255,.72)';
        a.style.background=isActive?'rgba(207,185,145,.07)':'';
        a.style.borderLeftColor=isActive?'#CFB991':'transparent';
      }});
      li2.appendChild(a);
      ul.appendChild(li2);
    }});

    ul.addEventListener('mouseenter',cancelTimer);
    ul.addEventListener('mouseleave',scheduleHide);
    _pdoc.body.appendChild(ul);
  }}

  window.navigate = function navigate(page){{
    window.parent.postMessage({{
      isStreamlitMessage:true,
      type:"streamlit:setComponentValue",
      value:page,
      dataType:"json"
    }},"*");
  }};

  /* ── Dark mode engine (pure JS + localStorage, no Python round-trip) ── */
  var DM_STYLE_ID = 'ec-dark-mode-css';
  var DM_KEY      = 'ec_dark_mode';
  var DM_CSS      = `{_DM_CSS}`;

  var DM_PLOT_DARK = {{
    paper_bgcolor: '#0f1117',
    plot_bgcolor:  '#1a1d27',
    'font.color':  '#e8e9ed',
    'xaxis.gridcolor':            '#2a2d3a',
    'xaxis.zerolinecolor':        '#2a2d3a',
    'xaxis.tickfont.color':       '#8890a1',
    'xaxis.rangeselector.bgcolor':'#1e2130',
    'xaxis.rangeselector.font.color':'#c8cdd8',
    'xaxis.rangeslider.bgcolor':  '#1a1d27',
    'yaxis.gridcolor':            '#2a2d3a',
    'yaxis.zerolinecolor':        '#2a2d3a',
    'yaxis.tickfont.color':       '#8890a1',
    'legend.font.color':          '#c8cdd8',
    'legend.bgcolor':             'rgba(15,17,23,0.5)'
  }};

  var DM_PLOT_LIGHT = {{
    paper_bgcolor: '#ffffff',
    plot_bgcolor:  '#fafaf8',
    'font.color':  '#000000',
    'xaxis.gridcolor':            '#EEEBE6',
    'xaxis.zerolinecolor':        '#EEEBE6',
    'xaxis.tickfont.color':       '#000000',
    'xaxis.rangeselector.bgcolor':'#f0ede8',
    'xaxis.rangeselector.font.color':'#000000',
    'xaxis.rangeslider.bgcolor':  '#ffffff',
    'yaxis.gridcolor':            '#EEEBE6',
    'yaxis.zerolinecolor':        '#EEEBE6',
    'yaxis.tickfont.color':       '#000000',
    'legend.font.color':          '#000000',
    'legend.bgcolor':             'rgba(255,255,255,0)'
  }};

  function _dmRelayoutAll(on) {{
    var pWin = window.parent;
    var pDoc = pWin.document;
    if (!pWin.Plotly) return;
    var upd = on ? DM_PLOT_DARK : DM_PLOT_LIGHT;
    pDoc.querySelectorAll('.js-plotly-plot').forEach(function(el) {{
      try {{ pWin.Plotly.relayout(el, upd); }} catch(e) {{}}
    }});
  }}

  var _dmObserver = null;
  var _dmObsTimer = null;

  function _dmStartObserver(on) {{
    if (_dmObserver) {{ _dmObserver.disconnect(); _dmObserver = null; }}
    if (!on) return;
    var pDoc = window.parent.document;
    _dmObserver = new window.parent.MutationObserver(function() {{
      clearTimeout(_dmObsTimer);
      _dmObsTimer = setTimeout(function() {{
        _dmRelayoutAll(true);
        _dmPatchInlineStyles(true);
        _dmStyleDataframes(true);
      }}, 300);
    }});
    _dmObserver.observe(pDoc.body, {{ childList: true, subtree: true }});
  }}

  /* ── Inline-style patch: CSS attr selectors fail because Chrome normalises
     hex colours to rgb() when parsing HTML. This JS checks computed
     backgroundColor and directly overwrites near-white inline backgrounds. ── */
  var DM_WHITE_TARGETS = [
    [255,255,255], [250,250,248], [255,253,245],
    [240,237,232], [245,242,238], [250,250,250], [248,248,248],
    [255,255,254], [254,254,254]
  ];

  /* Parse rgb/rgba string → [r,g,b] or null */
  function _dmParseRgb(s) {{
    var m = s.match(/rgba?\\(([\\d.]+),\\s*([\\d.]+),\\s*([\\d.]+)/);
    return m ? [parseInt(m[1]),parseInt(m[2]),parseInt(m[3])] : null;
  }}

  function _dmPatchInlineStyles(on) {{
    var pDoc = window.parent.document;
    if (on) {{
      pDoc.querySelectorAll('[style]').forEach(function(el) {{
        if (el.tagName==='CANVAS'||el.tagName==='IFRAME') return;
        if (el.closest && el.closest('.js-plotly-plot')) return;

        /* 1. BACKGROUND: near-white → dark */
        var bg = el.style.backgroundColor;
        if (bg && bg!=='' && bg!=='transparent' && bg!=='initial') {{
          var bc = _dmParseRgb(bg);
          if (bc && bc[0]>228 && bc[1]>228 && bc[2]>228) {{
            el.setAttribute('data-dm-bg', bg);
            el.style.setProperty('background','#1a1d27','important');
          }}
        }}

        /* 2. TEXT COLOR: dark gray / black → light
           Only patch achromatic/near-achromatic darks (all channels < 115).
           Saturated colours (green, red, gold) are kept — they have at
           least one channel well above the others. */
        var col = el.style.color;
        if (col && col!=='' && col!=='transparent' && col!=='inherit') {{
          var cc = _dmParseRgb(col);
          if (cc) {{
            var mx=Math.max(cc[0],cc[1],cc[2]);
            var mn=Math.min(cc[0],cc[1],cc[2]);
            /* near-achromatic dark: max < 115 AND saturation (max-min) < 40 */
            if (mx < 115 && (mx-mn) < 40) {{
              el.setAttribute('data-dm-col', col);
              var light = mx < 50 ? '#e8e9ed' : '#b8bec8';
              el.style.setProperty('color', light, 'important');
            }}
          }}
        }}

        /* 3. BORDER COLOR: near-white → Purdue gold (subtle) */
        var brd = el.style.borderColor ||
                  el.style.borderTopColor || el.style.borderRightColor ||
                  el.style.borderBottomColor || el.style.borderLeftColor;
        if (brd && brd!=='' && brd!=='transparent') {{
          var bv = _dmParseRgb(brd);
          if (bv && bv[0]>200 && bv[1]>195 && bv[2]>190) {{
            el.setAttribute('data-dm-brd', brd);
            /* use a muted gold for general borders, bright gold for accent */
            var isAccent = (el.style.borderLeftWidth==='3px'||el.style.borderLeftWidth==='4px'
                           ||el.style.borderTopWidth==='2px'||el.style.borderTopWidth==='3px');
            var goldBrd = isAccent ? '#CFB991' : 'rgba(207,185,145,0.35)';
            ['borderColor','borderTopColor','borderRightColor',
             'borderBottomColor','borderLeftColor'].forEach(function(p) {{
              if (el.style[p]) el.style.setProperty(
                p.replace(/([A-Z])/g,function(c){{return '-'+c.toLowerCase();}}),
                goldBrd, 'important');
            }});
          }}
        }}
      }});
    }} else {{
      ['data-dm-bg','data-dm-col','data-dm-brd'].forEach(function(attr) {{
        pDoc.querySelectorAll('['+attr+']').forEach(function(el) {{
          var orig = el.getAttribute(attr);
          if (attr==='data-dm-bg')  el.style.background  = orig;
          if (attr==='data-dm-col') el.style.color        = orig;
          if (attr==='data-dm-brd') el.style.borderColor  = orig;
          el.removeAttribute(attr);
        }});
      }});
    }}
  }}

  /* ── Dataframe iframe injection ──────────────────────────────────────────
     st.dataframe() renders inside a same-origin iframe.  CSS from the parent
     document cannot cross that iframe boundary, so we inject a <style> tag
     directly into each frame's document. Called after every relayout pass
     and from the MutationObserver so newly-rendered tables get styled too. */
  var DF_STYLE_ID = 'ec-df-dark';

  var DF_DARK_CSS = [
    'html,body{{background:#0f1117!important;color:#e8e9ed!important}}',
    '.dvn-scroller,.dvn-scroll-inner,.clip-region{{background:#0f1117!important}}',
    '[role="columnheader"],[role="rowheader"],.gdg-header,.gdg-group-header{{',
    '  background:#1a1d27!important;color:#CFB991!important;border-color:#2a2d3a!important}}',
    'table{{background:#0f1117!important;border-color:#2a2d3a!important}}',
    'thead,thead tr,thead th{{background:#1a1d27!important;color:#CFB991!important;border-color:#2a2d3a!important}}',
    'tbody tr{{background:#0f1117!important;color:#e8e9ed!important}}',
    'tbody tr:nth-child(even){{background:#141720!important}}',
    'tbody tr:hover{{background:#1e2130!important}}',
    'td,th{{color:#e8e9ed!important;border-color:#2a2d3a!important}}',
    '[data-testid="stFullScreenFrame"]{{background:#0f1117!important}}',
    'button,select,input{{background:#1a1d27!important;color:#e8e9ed!important;border-color:#2a2d3a!important}}',
    'div[style*="rgb(255"]{{background:#0f1117!important;color:#e8e9ed!important}}',
    'div[style*="rgb(250"]{{background:#161920!important;color:#e8e9ed!important}}',
    'span[style*="color: rgb(0"]{{color:#e8e9ed!important}}',
    'span[style*="color: rgb(5"]{{color:#e8e9ed!important}}',
  ].join('');

  function _dmStyleDataframes(on) {{
    var pDoc = window.parent.document;
    /* target both stDataFrame wrappers and any inner iframes */
    pDoc.querySelectorAll('[data-testid="stDataFrame"] iframe, [data-testid="stDataFrame"] > div iframe').forEach(function(fr) {{
      try {{
        var fd = fr.contentDocument || (fr.contentWindow && fr.contentWindow.document);
        if (!fd || !fd.head) return;
        var ex = fd.getElementById(DF_STYLE_ID);
        if (on) {{
          if (!ex) {{
            var s = fd.createElement('style');
            s.id = DF_STYLE_ID;
            s.textContent = DF_DARK_CSS;
            fd.head.appendChild(s);
          }}
          /* also patch inline bg/col inside the frame */
          fd.querySelectorAll('[style]').forEach(function(el) {{
            var bg = el.style.backgroundColor;
            if (bg) {{
              var bc = _dmParseRgb(bg);
              if (bc && bc[0]>228 && bc[1]>228 && bc[2]>228)
                el.style.setProperty('background','#0f1117','important');
            }}
            var col = el.style.color;
            if (col) {{
              var cc = _dmParseRgb(col);
              if (cc && Math.max(cc[0],cc[1],cc[2])<115 && (Math.max(cc[0],cc[1],cc[2])-Math.min(cc[0],cc[1],cc[2]))<40)
                el.style.setProperty('color','#e8e9ed','important');
            }}
          }});
        }} else {{
          if (ex) ex.remove();
        }}
      }} catch(e) {{}}
    }});
  }}

  function _dmApply(on) {{
    var pDoc = window.parent.document;
    /* CSS layer */
    var existing = pDoc.getElementById(DM_STYLE_ID);
    if (on) {{
      if (!existing) {{
        var s = pDoc.createElement('style');
        s.id = DM_STYLE_ID;
        s.textContent = DM_CSS;
        pDoc.head.appendChild(s);
      }}
    }} else {{
      if (existing) existing.remove();
    }}
    /* Plotly charts */
    _dmRelayoutAll(on);
    /* Inline-style white-patch fix (main doc) */
    setTimeout(function() {{ _dmPatchInlineStyles(on); }}, 150);
    /* Dataframe iframes */
    setTimeout(function() {{ _dmStyleDataframes(on); }}, 300);
    /* Observer for future charts + new elements */
    _dmStartObserver(on);
    /* button icon */
    var btn = document.getElementById('dm-btn');
    if (btn) btn.innerHTML = on ? '&#x2600;' : '&#x1F319;';
  }}

  window.ecToggleDark = function() {{
    var isDark = !!window.parent.document.getElementById(DM_STYLE_ID);
    var next = !isDark;
    try {{ localStorage.setItem(DM_KEY, next ? '1' : '0'); }} catch(e) {{}}
    _dmApply(next);
  }};

  document.addEventListener('DOMContentLoaded',function(){{
    /* Restore dark mode from localStorage on every load */
    try {{
      if (localStorage.getItem(DM_KEY) === '1') {{
        /* Two passes: first CSS+Plotly at 400ms, then inline-style patch at 900ms
           (Streamlit may still be mounting components at 400ms) */
        setTimeout(function() {{ _dmApply(true); }}, 400);
        setTimeout(function() {{ _dmPatchInlineStyles(true); }}, 900);
        setTimeout(function() {{ _dmStyleDataframes(true); }}, 1200);
      }}
    }} catch(e) {{}}

    /* Wire hover items */
    document.querySelectorAll('li.ni').forEach(function(li){{
      if(!li.querySelector('ul.drop')) return;
      li.addEventListener('mouseenter',function(){{ buildDropdown(li); }});
      li.addEventListener('mouseleave',scheduleHide);
    }});

    /* Mark active page + group */
    var cur={json.dumps(current)};
    var ANALYSIS=['war_impact_map','geopolitical','correlation','spillover','watchlist'];
    var STRATEGY=['trade_ideas','stress_test'];
    var RESEARCH=['model_accuracy','ai_chat'];
    var ABOUT   =['about_heramb','about_jiahe','about_ilian'];
    document.querySelectorAll('[data-pg]').forEach(function(a){{
      if(a.dataset.pg===cur) a.classList.add('active');
    }});
    var MAP={{'ga':ANALYSIS,'gs':STRATEGY,'gr':RESEARCH,'gab':ABOUT}};
    Object.keys(MAP).forEach(function(id){{
      if(MAP[id].indexOf(cur)>-1){{
        var el=document.getElementById(id);
        if(el) el.classList.add('hd-active');
      }}
    }});
  }});
}})();
</script>

<div id="nav">
  <!-- Logotype -->
  <a class="brand" href="#" onclick="navigate('overview');return false;">
    <div class="brand-mark">
      <span class="mk-top">E&amp;C</span>
      <span class="mk-bot">MON</span>
    </div>
    <div class="brand-div"></div>
    <div class="brand-text">
      <span class="bm">Equity <em>&amp;</em> Commodities Spillover</span>
      <span class="bs">Purdue Daniels &middot; MGMT&nbsp;69000&ndash;120</span>
    </div>
  </a>

  <ul class="links">
    <li class="ni">
      <a class="lnk {'active' if current == 'overview' else ''}" data-pg="overview"
         href="#" onclick="navigate('overview');return false;">Overview</a>
    </li>

    <li class="ni" id="ga">
      <span class="lnk">Analysis <span class="ct">&#9660;</span></span>
      <ul class="drop">
        <li><a data-pg="war_impact_map" href="?page=war_impact_map" target="_parent" class="{'active' if current=='war_impact_map' else ''}">War Impact Map</a></li>
        <li><a data-pg="geopolitical"   href="?page=geopolitical"   target="_parent" class="{'active' if current=='geopolitical' else ''}">Geopolitical Triggers</a></li>
        <li><a data-pg="correlation"    href="?page=correlation"    target="_parent" class="{'active' if current=='correlation' else ''}">Correlation Analysis</a></li>
        <li><a data-pg="spillover"      href="?page=spillover"      target="_parent" class="{'active' if current=='spillover' else ''}">Spillover Analytics</a></li>
        <li><a data-pg="watchlist"      href="?page=watchlist"      target="_parent" class="{'active' if current=='watchlist' else ''}">Commodities to Watch</a></li>
      </ul>
    </li>

    <li class="ni" id="gs">
      <span class="lnk">Strategy <span class="ct">&#9660;</span></span>
      <ul class="drop">
        <li><a data-pg="trade_ideas" href="?page=trade_ideas" target="_parent" class="{'active' if current=='trade_ideas' else ''}">Trade Ideas</a></li>
        <li><a data-pg="stress_test" href="?page=stress_test" target="_parent" class="{'active' if current=='stress_test' else ''}">Portfolio Stress Test</a></li>
      </ul>
    </li>

    <li class="ni" id="gr">
      <span class="lnk">Research <span class="ct">&#9660;</span></span>
      <ul class="drop">
        <li><a data-pg="model_accuracy" href="?page=model_accuracy" target="_parent" class="{'active' if current=='model_accuracy' else ''}">Performance Review</a></li>
        <li><a data-pg="ai_chat"        href="?page=ai_chat"        target="_parent" class="{'active' if current=='ai_chat' else ''}">AI Analyst</a></li>
      </ul>
    </li>

    <li class="ni" id="gab">
      <span class="lnk">About <span class="ct">&#9660;</span></span>
      <ul class="drop">
        <li><a data-pg="about_heramb" href="?page=about_heramb" target="_parent" class="{'active' if current=='about_heramb' else ''}">Heramb S. Patkar</a></li>
        <li><a data-pg="about_jiahe"  href="?page=about_jiahe"  target="_parent" class="{'active' if current=='about_jiahe' else ''}">Jiahe Miao</a></li>
        <li><a data-pg="about_ilian"  href="?page=about_ilian"  target="_parent" class="{'active' if current=='about_ilian' else ''}">Ilian Zalomai</a></li>
      </ul>
    </li>
  </ul>

  <div class="ds">
    <div class="dsi"><span class="dot" style="color:#CFB991">&#9679;</span><span style="color:rgba(255,255,255,.40)">Yahoo Finance</span></div>
    <div class="dsi"><span class="dot" style="color:{_fred_col}">{_fred_dot}</span><span style="color:{_fred_col}">FRED</span></div>
    <div class="dsi"><span class="dot" style="color:{_fd_col}">{_fd_dot}</span><span style="color:{_fd_col}">FinancialDatasets</span></div>
    <a id="dm-btn" class="dm-toggle" href="#" onclick="ecToggleDark();return false;" title="Toggle dark mode">&#x1F319;</a>
  </div>
</div>
</body></html>"""

_nav_click = components.html(_NAVBAR, height=1, scrolling=False)
_nav_last  = st.session_state.get("_nav_last", "")

if _nav_click and _nav_click != _nav_last:
    st.session_state["_nav_last"] = _nav_click
    if _nav_click in _VALID_PAGES:
        st.session_state["current_page"] = _nav_click
        current = _nav_click
        st.rerun()

# ── Date range strip (hidden on About pages — they need no data range) ────────
_is_about = current in _ABOUT_PAGES

if not _is_about:
    _dc1, _dc2, _dc3 = st.columns([1, 1, 5])
    start_date = _dc1.date_input(
        "From", value=date(2010, 1, 1),
        min_value=date(2000, 1, 1), max_value=date.today(), key="g_start",
    )
    end_date = _dc2.date_input(
        "To", value=date.today(),
        min_value=date(2000, 1, 1), max_value=date.today(), key="g_end",
    )
    st.markdown('<div style="border-top:1px solid #E8E5E0;margin:0.4rem 0 1.4rem"></div>',
                unsafe_allow_html=True)
else:
    # Provide dummy dates so the router lambdas below don't fail
    start_date = st.session_state.get("g_start", date(2010, 1, 1))
    end_date   = st.session_state.get("g_end",   date.today())

# ── Router ────────────────────────────────────────────────────────────────────
from src.pages.overview        import page_overview
from src.pages.war_impact_map  import page_war_impact_map
from src.pages.geopolitical    import page_geopolitical
from src.pages.correlation     import page_correlation
from src.pages.spillover       import page_spillover
from src.pages.watchlist       import page_watchlist
from src.pages.trade_ideas     import page_trade_ideas
from src.pages.stress_test     import page_stress_test
from src.pages.model_accuracy  import page_model_accuracy
from src.pages.ai_chat         import page_ai_chat
from src.pages.about_heramb    import page_about_heramb
from src.pages.about_jiahe     import page_about_jiahe
from src.pages.about_ilian     import page_about_ilian

_start = str(start_date)
_end   = str(end_date)

_PAGE_MAP = {
    "overview":       lambda: page_overview(_start, _end, _FRED_KEY),
    "war_impact_map": lambda: page_war_impact_map(_start, _end, _FRED_KEY),
    "geopolitical":   lambda: page_geopolitical(_start, _end, _FRED_KEY),
    "correlation":    lambda: page_correlation(_start, _end, _FRED_KEY),
    "spillover":      lambda: page_spillover(_start, _end, _FRED_KEY),
    "watchlist":      lambda: page_watchlist(_start, _end, _FRED_KEY),
    "trade_ideas":    lambda: page_trade_ideas(_start, _end, _FRED_KEY),
    "stress_test":    lambda: page_stress_test(_start, _end, _FRED_KEY),
    "model_accuracy": lambda: page_model_accuracy(_start, _end, _FRED_KEY),
    "ai_chat":        lambda: page_ai_chat(_start, _end),
    "about_heramb":   page_about_heramb,
    "about_jiahe":    page_about_jiahe,
    "about_ilian":    page_about_ilian,
}

_PAGE_MAP.get(current, _PAGE_MAP["overview"])()
